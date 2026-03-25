package main

import (
	"bytes"
	"crypto/sha256"
	"database/sql"
	"embed"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"io"
	"io/fs"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	_ "github.com/duckdb/duckdb-go/v2"
	"github.com/gin-gonic/gin"
	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

var (
	//go:embed frontend/build
	frontendBuildEmbed embed.FS

	// Standard ImageNet normalization used by timm resnet models.
	mean = []float32{0.485, 0.456, 0.406}
	std  = []float32{0.229, 0.224, 0.225}

	acceptedFileTypes = map[string]struct{}{
		".jpeg": {}, ".jpg": {}, ".png": {}, ".gif": {}, ".tiff": {}, ".tif": {}, ".bmp": {}, ".webp": {},
	}

	waterfallCollection = "waterfall"
	fftCollection       = "fft"
	waterfallPath       = "datasets/waterfall"
	fftPath             = "datasets/fft"

	knownFrequencies map[string]interface{}

	db              *sql.DB
	embeddingStatus atomic.Value

	extractor *Extractor
)

type Extractor struct {
	modelPath string
	mu        sync.Mutex
}

func newExtractor(modelPath string) *Extractor {
	return &Extractor{modelPath: modelPath}
}

// preprocessImage converts image to grayscale->RGB, resizes to 224x224,
// and returns a NCHW float32 tensor normalized with ImageNet mean/std.
func preprocessImage(img image.Image, outH, outW int) []float32 {
	bounds := img.Bounds()
	gray := image.NewGray(bounds)
	draw.Draw(gray, bounds, img, bounds.Min, draw.Src)

	rgb := image.NewRGBA(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			g := gray.GrayAt(x, y).Y
			c := color.RGBA{g, g, g, 255}
			rgb.Set(x, y, c)
		}
	}

	resized := resize.Resize(uint(outW), uint(outH), rgb, resize.Bilinear)
	data := make([]float32, 3*outH*outW)
	idx := 0
	for c := 0; c < 3; c++ {
		for y := 0; y < outH; y++ {
			for x := 0; x < outW; x++ {
				r, g, b, _ := resized.At(x, y).RGBA()
				rf := float32(r) / 65535.0
				gf := float32(g) / 65535.0
				bf := float32(b) / 65535.0

				var v float32
				switch c {
				case 0:
					v = (rf - mean[0]) / std[0]
				case 1:
					v = (gf - mean[1]) / std[1]
				case 2:
					v = (bf - mean[2]) / std[2]
				}
				data[idx] = v
				idx++
			}
		}
	}
	return data
}

func (e *Extractor) EmbedImage(img image.Image) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	inputData := preprocessImage(img, 224, 224)
	inputShape := ort.NewShape(1, 3, 224, 224)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputShape := ort.NewShape(1, 512)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(
		e.modelPath,
		[]string{"input"},
		[]string{"features"},
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		return nil, err
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, err
	}

	out := outputTensor.GetData()
	var sum float64
	for _, v := range out {
		sum += float64(v * v)
	}
	norm := float32(math.Sqrt(sum))
	if norm > 0 {
		for i := range out {
			out[i] /= norm
		}
	}

	copyOut := make([]float32, len(out))
	copy(copyOut, out)
	return copyOut, nil
}

func fileToImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func imageToBase64(img image.Image) (string, error) {
	buf := bytes.NewBuffer(nil)
	if err := png.Encode(buf, img); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

func drawText(img draw.Image, x, y int, text string) {
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.White),
		Face: basicfont.Face7x13,
		Dot:  fixed.P(x, y),
	}
	d.DrawString(text)
}

func getModelPath() (string, error) {
	if _, err := os.Stat("RadioNet.onnx"); err == nil {
		return "RadioNet.onnx", nil
	}
	if _, err := os.Stat("RadioNet/RadioNet.onnx"); err == nil {
		return "RadioNet/RadioNet.onnx", nil
	}
	return "", fmt.Errorf("failed to find model file: tried RadioNet.onnx and RadioNet/RadioNet.onnx")
}

func initFrequencies() error {
	f, err := os.Open("known_frequencies.json")
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewDecoder(f).Decode(&knownFrequencies)
}

func initDuckDB() error {
	var err error
	db, err = sql.Open("duckdb", "classidyne.db")
	if err != nil {
		return err
	}

	stmts := []string{
		"INSTALL vss;",
		"LOAD vss;",
		"PRAGMA hnsw_enable_experimental_persistence = true;",
		"SET hnsw_enable_experimental_persistence = true;",
		`CREATE TABLE IF NOT EXISTS waterfall (
			vector FLOAT[512],
			filepath VARCHAR,
			filehash VARCHAR,
			class VARCHAR
		);`,
		`CREATE TABLE IF NOT EXISTS fft (
			vector FLOAT[512],
			filepath VARCHAR,
			filehash VARCHAR,
			class VARCHAR
		);`,
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_waterfall_filehash ON waterfall(filehash);",
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_fft_filehash ON fft(filehash);",
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_waterfall_filepath ON waterfall(filepath);",
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_fft_filepath ON fft(filepath);",
	}

	for _, stmt := range stmts {
		if _, err := db.Exec(stmt); err != nil {
			return fmt.Errorf("duckdb init failed on %q: %w", stmt, err)
		}
	}

	vectorIndexStmts := []string{
		"CREATE INDEX IF NOT EXISTS idx_waterfall_vec ON waterfall USING HNSW (vector) WITH (metric='cosine');",
		"CREATE INDEX IF NOT EXISTS idx_fft_vec ON fft USING HNSW (vector) WITH (metric='cosine');",
	}

	for _, stmt := range vectorIndexStmts {
		if _, err := db.Exec(stmt); err != nil {
			// Allow startup without ANN indexes when persistent HNSW is unavailable.
			if strings.Contains(err.Error(), "hnsw_enable_experimental_persistence") {
				log.Printf("warning: skipping HNSW index creation: %v", err)
				continue
			}
			return fmt.Errorf("duckdb init failed on %q: %w", stmt, err)
		}
	}
	return nil
}

func computeSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func classifyFromPath(path string) ([]float32, error) {
	img, err := fileToImage(path)
	if err != nil {
		return nil, err
	}
	return extractor.EmbedImage(img)
}

func existsByHash(collection, hash string) (bool, error) {
	var count int
	err := db.QueryRow("SELECT COUNT(*) FROM "+collection+" WHERE filehash = ?", hash).Scan(&count)
	if err != nil {
		return false, err
	}
	return count > 0, nil
}

func insertBatch(collection string, rows []map[string]interface{}) int {
	tx, err := db.Begin()
	if err != nil {
		log.Printf("insert batch begin failed: %v", err)
		return 0
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare("INSERT INTO " + collection + " (vector, filepath, filehash, class) VALUES (?, ?, ?, ?)")
	if err != nil {
		log.Printf("insert batch prepare failed: %v", err)
		return 0
	}
	defer stmt.Close()

	inserted := 0
	for _, row := range rows {
		_, err := stmt.Exec(row["vector"], row["filepath"], row["filehash"], row["class"])
		if err != nil {
			continue
		}
		inserted++
	}

	if err := tx.Commit(); err != nil {
		log.Printf("insert batch commit failed: %v", err)
		return 0
	}
	return inserted
}

func embedDataset(folderPath, collection string) int {
	const batchSize = 500
	batchData := make([]map[string]interface{}, 0, batchSize)
	inserted := 0

	_ = filepath.WalkDir(folderPath, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(d.Name()))
		if _, ok := acceptedFileTypes[ext]; !ok {
			return nil
		}

		filehash, err := computeSHA256(path)
		if err != nil {
			log.Printf("failed hash for %s: %v", path, err)
			return nil
		}

		exists, err := existsByHash(collection, filehash)
		if err == nil && exists {
			return nil
		}

		emb, err := classifyFromPath(path)
		if err != nil {
			log.Printf("failed embedding for %s: %v", path, err)
			return nil
		}

		batchData = append(batchData, map[string]interface{}{
			"vector":   emb,
			"filepath": path,
			"filehash": filehash,
			"class":    filepath.Base(filepath.Dir(path)),
		})

		if len(batchData) >= batchSize {
			inserted += insertBatch(collection, batchData)
			batchData = batchData[:0]
		}
		return nil
	})

	if len(batchData) > 0 {
		inserted += insertBatch(collection, batchData)
	}

	return inserted
}

func embedAllDatasets() {
	defer func() {
		if r := recover(); r != nil {
			embeddingStatus.Store("Fatal Error")
			log.Printf("embedding panic: %v", r)
		}
	}()

	embeddingStatus.Store("Processing Waterfall Images")
	waterfallInserted := embedDataset(waterfallPath, waterfallCollection)

	embeddingStatus.Store("Processing FFT Images")
	fftInserted := embedDataset(fftPath, fftCollection)

	embeddingStatus.Store("Finished")
	log.Printf("inserted %d waterfall and %d fft entries", waterfallInserted, fftInserted)
}

func formatFrequency(freq float64) string {
	if freq >= 1e9 {
		return fmt.Sprintf("%.3f GHz", freq/1e9)
	}
	if freq >= 1e6 {
		return fmt.Sprintf("%.3f MHz", freq/1e6)
	}
	return fmt.Sprintf("%.0f Hz", freq)
}

func toFloat64Slice(v interface{}) ([]float64, bool) {
	arr, ok := v.([]interface{})
	if !ok {
		return nil, false
	}
	out := make([]float64, 0, len(arr))
	for _, elem := range arr {
		num, ok := elem.(float64)
		if !ok {
			return nil, false
		}
		out = append(out, num)
	}
	return out, true
}

func getFrequencyRange(signalClass string) string {
	raw, ok := knownFrequencies[signalClass]
	if !ok || raw == nil {
		return "NA"
	}

	if simple, ok := toFloat64Slice(raw); ok && len(simple) == 2 {
		if simple[0] == simple[1] {
			return formatFrequency(simple[0])
		}
		return fmt.Sprintf("%s - %s", formatFrequency(simple[0]), formatFrequency(simple[1]))
	}

	if ranges, ok := raw.([]interface{}); ok {
		parts := make([]string, 0, len(ranges))
		for _, r := range ranges {
			if pair, ok := toFloat64Slice(r); ok && len(pair) == 2 {
				if pair[0] == pair[1] {
					parts = append(parts, formatFrequency(pair[0]))
				} else {
					parts = append(parts, fmt.Sprintf("%s - %s", formatFrequency(pair[0]), formatFrequency(pair[1])))
				}
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, ", ")
		}
	}

	if bands, ok := raw.(map[string]interface{}); ok {
		keys := make([]string, 0, len(bands))
		for k := range bands {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		lines := make([]string, 0, len(keys))
		for _, band := range keys {
			v := bands[band]
			if pair, ok := toFloat64Slice(v); ok && len(pair) == 2 {
				if pair[0] == pair[1] {
					lines = append(lines, fmt.Sprintf("- %s: %s", band, formatFrequency(pair[0])))
				} else {
					lines = append(lines, fmt.Sprintf("- %s: %s - %s", band, formatFrequency(pair[0]), formatFrequency(pair[1])))
				}
				continue
			}

			if rangeList, ok := v.([]interface{}); ok {
				parts := []string{}
				for _, r := range rangeList {
					if pair, ok := toFloat64Slice(r); ok && len(pair) == 2 {
						if pair[0] == pair[1] {
							parts = append(parts, formatFrequency(pair[0]))
						} else {
							parts = append(parts, fmt.Sprintf("%s - %s", formatFrequency(pair[0]), formatFrequency(pair[1])))
						}
					}
				}
				if len(parts) > 0 {
					lines = append(lines, fmt.Sprintf("- %s: %s", band, strings.Join(parts, ", ")))
				}
			}
		}
		if len(lines) > 0 {
			return strings.Join(lines, "\n")
		}
	}

	return "Unknown"
}

type SearchHit struct {
	Filepath   string
	Filehash   string
	Class      string
	Similarity float64
}

func vectorSearch(collection string, embedding []float32, limit int) ([]SearchHit, error) {
	query := `
		SELECT filepath, filehash, class,
		       array_cosine_similarity(vector, ?::FLOAT[512]) AS similarity
		FROM ` + collection + `
		ORDER BY similarity DESC
		LIMIT ?;
	`

	rows, err := db.Query(query, embedding, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	hits := make([]SearchHit, 0, limit)
	for rows.Next() {
		var h SearchHit
		if err := rows.Scan(&h.Filepath, &h.Filehash, &h.Class, &h.Similarity); err != nil {
			continue
		}
		hits = append(hits, h)
	}
	return hits, rows.Err()
}

func getTableCount(collection string) (int64, error) {
	var c int64
	err := db.QueryRow("SELECT COUNT(*) FROM " + collection).Scan(&c)
	return c, err
}

func listTypes(basePath string) ([]string, error) {
	entries, err := os.ReadDir(basePath)
	if err != nil {
		return nil, err
	}
	types := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			types = append(types, e.Name())
		}
	}
	sort.Strings(types)
	return types, nil
}

func createTypeCollage(collection, signalType string) (string, error) {
	fullPath := filepath.Join("datasets", collection, signalType)
	entries, err := os.ReadDir(fullPath)
	if err != nil {
		return "", err
	}

	canvas := image.NewRGBA(image.Rect(0, 0, 150*5, 150*5))
	idx := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if _, ok := acceptedFileTypes[ext]; !ok {
			continue
		}
		img, err := fileToImage(filepath.Join(fullPath, entry.Name()))
		if err != nil {
			continue
		}
		resized := resize.Resize(150, 150, img, resize.Bilinear)
		x := (idx % 5) * 150
		y := (idx / 5) * 150
		draw.Draw(canvas, image.Rect(x, y, x+150, y+150), resized, image.Point{}, draw.Src)
		idx++
		if idx >= 25 {
			break
		}
	}

	if idx == 0 {
		return "", fmt.Errorf("no images found for selected type")
	}

	return imageToBase64(canvas)
}

func setupRouter() *gin.Engine {
	r := gin.Default()

	frontendFS, err := fs.Sub(frontendBuildEmbed, "frontend/build")
	if err != nil {
		log.Fatalf("failed to open embedded frontend build files: %v", err)
	}
	frontendFileServer := http.FileServer(http.FS(frontendFS))

	r.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	})

	api := r.Group("/api")
	{
		api.POST("/classify", func(c *gin.Context) {
			collection := c.PostForm("collection")
			if collection != waterfallCollection && collection != fftCollection {
				c.JSON(http.StatusBadRequest, gin.H{"success": false, "message": "collection must be waterfall or fft", "class_scores": []interface{}{}, "collage_image": nil})
				return
			}

			threshold := 0.5
			if raw := c.PostForm("similarity_threshold"); raw != "" {
				if parsed, err := strconv.ParseFloat(raw, 64); err == nil {
					threshold = parsed
				}
			}

			file, err := c.FormFile("query_image")
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"success": false, "message": "query_image is required", "class_scores": []interface{}{}, "collage_image": nil})
				return
			}

			rc, err := file.Open()
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"success": false, "message": fmt.Sprintf("failed to open image: %v", err), "class_scores": []interface{}{}, "collage_image": nil})
				return
			}
			defer rc.Close()

			img, _, err := image.Decode(rc)
			if err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"success": false, "message": fmt.Sprintf("failed to decode image: %v", err), "class_scores": []interface{}{}, "collage_image": nil})
				return
			}

			emb, err := extractor.EmbedImage(img)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"success": false, "message": fmt.Sprintf("embedding failed: %v", err), "class_scores": []interface{}{}, "collage_image": nil})
				return
			}

			hits, err := vectorSearch(collection, emb, 20)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"success": false, "message": fmt.Sprintf("search failed: %v", err), "class_scores": []interface{}{}, "collage_image": nil})
				return
			}

			canvas := image.NewRGBA(image.Rect(0, 0, 150*5, 150*4))
			classCounts := map[string]int{}
			idx := 0
			for _, hit := range hits {
				if hit.Similarity < threshold || idx >= 20 {
					continue
				}
				img2, err := fileToImage(hit.Filepath)
				if err != nil {
					continue
				}
				thumb := resize.Resize(150, 150, img2, resize.Bilinear)
				x := (idx % 5) * 150
				y := (idx / 5) * 150
				draw.Draw(canvas, image.Rect(x, y, x+150, y+150), thumb, image.Point{}, draw.Src)
				drawText(canvas, x+4, y+125, "Class: "+hit.Class)
				drawText(canvas, x+4, y+142, fmt.Sprintf("Score: %.2f", hit.Similarity))
				classCounts[hit.Class]++
				idx++
			}

			total := 0
			for _, v := range classCounts {
				total += v
			}

			type classScore struct {
				Class          string  `json:"class"`
				Confidence     float64 `json:"confidence"`
				FrequencyRange string  `json:"frequency_range"`
			}
			scores := make([]classScore, 0, len(classCounts))
			if total > 0 {
				for cls, count := range classCounts {
					scores = append(scores, classScore{
						Class:          cls,
						Confidence:     float64(count) / float64(total) * 100.0,
						FrequencyRange: getFrequencyRange(cls),
					})
				}
				sort.Slice(scores, func(i, j int) bool { return scores[i].Confidence > scores[j].Confidence })
			}

			var collage interface{} = nil
			if total > 0 {
				if b64, err := imageToBase64(canvas); err == nil {
					collage = b64
				}
			}

			c.JSON(http.StatusOK, gin.H{
				"success":       true,
				"message":       fmt.Sprintf("Found %d images with similarity score above %.2f.", total, threshold),
				"class_scores":  scores,
				"collage_image": collage,
			})
		})

		api.POST("/start-embedding", func(c *gin.Context) {
			status := embeddingStatus.Load().(string)
			if status == "Idle" || status == "Finished" || status == "Fatal Error" {
				embeddingStatus.Store("Starting")
				go embedAllDatasets()
				c.JSON(http.StatusOK, gin.H{"success": true, "message": "Embedding task started in the background."})
				return
			}
			c.JSON(http.StatusOK, gin.H{"success": false, "message": fmt.Sprintf("Embedding task is already running. Current status: %s", status)})
		})

		api.GET("/stats", func(c *gin.Context) {
			waterfallCount, err1 := getTableCount(waterfallCollection)
			fftCount, err2 := getTableCount(fftCollection)
			if err1 != nil || err2 != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"success": false, "message": "Error fetching stats"})
				return
			}
			c.JSON(http.StatusOK, gin.H{
				"success":          true,
				"message":          "Stats fetched successfully.",
				"embedding_status": embeddingStatus.Load().(string),
				"waterfall_size":   waterfallCount,
				"fft_size":         fftCount,
			})
		})

		api.DELETE("/delete_image", func(c *gin.Context) {
			identifier := c.Query("identifier")
			if identifier == "" {
				c.JSON(http.StatusBadRequest, gin.H{"success": false, "message": "identifier query param is required"})
				return
			}

			deleted := int64(0)
			for _, table := range []string{waterfallCollection, fftCollection} {
				res, err := db.Exec("DELETE FROM "+table+" WHERE filehash = ?", identifier)
				if err == nil {
					if n, err := res.RowsAffected(); err == nil {
						deleted += n
					}
				}
			}

			filename := filepath.Base(identifier)
			for _, table := range []string{waterfallCollection, fftCollection} {
				res, err := db.Exec("DELETE FROM "+table+" WHERE filepath LIKE ?", "%/"+filename)
				if err == nil {
					if n, err := res.RowsAffected(); err == nil {
						deleted += n
					}
				}
			}

			if deleted > 0 {
				c.JSON(http.StatusOK, gin.H{"success": true, "message": fmt.Sprintf("Deleted %d images with identifier '%s'.", deleted, identifier)})
				return
			}
			c.JSON(http.StatusNotFound, gin.H{"success": false, "message": "No images found with the given identifier."})
		})

		api.GET("/find_image", func(c *gin.Context) {
			identifier := c.Query("identifier")
			if identifier == "" {
				c.JSON(http.StatusBadRequest, gin.H{"success": false, "message": "identifier query param is required"})
				return
			}

			queryOne := func(table, where string, arg string) (bool, map[string]interface{}) {
				row := db.QueryRow("SELECT filepath, filehash, class FROM "+table+" WHERE "+where+" LIMIT 1", arg)
				var filepathV, filehashV, classV string
				if err := row.Scan(&filepathV, &filehashV, &classV); err != nil {
					return false, nil
				}
				img, err := fileToImage(filepathV)
				if err != nil {
					return false, nil
				}
				thumb := resize.Resize(150, 150, img, resize.Bilinear)
				b64, err := imageToBase64(thumb)
				if err != nil {
					return false, nil
				}
				return true, map[string]interface{}{
					"success":  true,
					"message":  "Found a matching image.",
					"filepath": filepathV,
					"filehash": filehashV,
					"class":    classV,
					"image":    b64,
				}
			}

			for _, table := range []string{waterfallCollection, fftCollection} {
				if ok, payload := queryOne(table, "filehash = ?", identifier); ok {
					c.JSON(http.StatusOK, payload)
					return
				}
			}

			filename := filepath.Base(identifier)
			for _, table := range []string{waterfallCollection, fftCollection} {
				if ok, payload := queryOne(table, "filepath LIKE ?", "%/"+filename); ok {
					c.JSON(http.StatusOK, payload)
					return
				}
			}

			c.JSON(http.StatusNotFound, gin.H{"success": false, "message": "No matching images found."})
		})

		api.GET("/waterfall_types", func(c *gin.Context) {
			types, err := listTypes(waterfallPath)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"success": false, "message": fmt.Sprintf("Error listing waterfall types: %v", err)})
				return
			}
			c.JSON(http.StatusOK, gin.H{"success": true, "message": "Waterfall types listed successfully.", "types": types})
		})

		api.GET("/fft_types", func(c *gin.Context) {
			types, err := listTypes(fftPath)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"success": false, "message": fmt.Sprintf("Error listing FFT types: %v", err)})
				return
			}
			c.JSON(http.StatusOK, gin.H{"success": true, "message": "FFT types listed successfully.", "types": types})
		})

		api.GET("/type_collage", func(c *gin.Context) {
			signalType := c.Query("type")
			collection := c.Query("collection")
			if signalType == "" || (collection != waterfallCollection && collection != fftCollection) {
				c.JSON(http.StatusBadRequest, gin.H{"success": false, "message": "collection and type are required"})
				return
			}

			collage, err := createTypeCollage(collection, signalType)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"success": false, "message": fmt.Sprintf("Error generating collage: %v", err)})
				return
			}

			c.JSON(http.StatusOK, gin.H{"success": true, "message": "Collage generated successfully.", "collage_base64": collage})
		})
	}

	r.NoRoute(func(c *gin.Context) {
		requestPath := c.Request.URL.Path
		if requestPath == "/api" || strings.HasPrefix(requestPath, "/api/") {
			c.JSON(http.StatusNotFound, gin.H{"success": false, "message": "Not Found"})
			return
		}

		if _, err := fs.Stat(frontendFS, strings.TrimPrefix(requestPath, "/")); err == nil {
			frontendFileServer.ServeHTTP(c.Writer, c.Request)
			return
		}

		c.Request.URL.Path = "/"
		frontendFileServer.ServeHTTP(c.Writer, c.Request)
	})
	return r
}

func main() {
	ortLibPath := os.Getenv("ORT_SHARED_LIB")
	if ortLibPath == "" {
		ortLibPath = "/opt/homebrew/lib/libonnxruntime.dylib"
	}
	ort.SetSharedLibraryPath(ortLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		log.Fatalf("failed to init ONNX Runtime: %v", err)
	}
	defer func() {
		if err := ort.DestroyEnvironment(); err != nil {
			log.Printf("failed to destroy ONNX runtime environment: %v", err)
		}
	}()

	modelPath, err := getModelPath()
	if err != nil {
		log.Fatal(err)
	}
	extractor = newExtractor(modelPath)

	if err := initFrequencies(); err != nil {
		log.Fatalf("failed to load known frequencies: %v", err)
	}

	if err := initDuckDB(); err != nil {
		log.Fatalf("failed to initialize duckdb: %v", err)
	}
	defer db.Close()

	embeddingStatus.Store("Idle")

	router := setupRouter()

	host := "0.0.0.0"
	startPort := 5000
	endPort := 5005

	for p := startPort; p <= endPort; p++ {
		addr := fmt.Sprintf("%s:%d", host, p)
		ln, err := net.Listen("tcp", addr)
		if err != nil {
			log.Printf("Port %d unavailable, trying next...", p)
			continue
		}
		_ = ln.Close()
		log.Printf("Starting server on port %d", p)
		if err := router.Run(addr); err != nil {
			log.Fatalf("server failed: %v", err)
		}
		os.Exit(1)
	}

	log.Fatalf("No ports available between %d and %d. Server aborting.", startPort, endPort)
}
