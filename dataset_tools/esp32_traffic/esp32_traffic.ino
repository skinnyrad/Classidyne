// Classidyne dataset v3: 2.4 GHz traffic source for real Wi-Fi / Bluetooth LE waterfall captures.
//
// The ESP32 transmits at low power on a fixed channel so a HackRF in SDR++ (10-20 MS/s span)
// sees dense, realistic bursts:
//   * Wi-Fi: a soft-AP beacon ("CLASSIDYNE-TEST") plus raw broadcast data frames of random
//     length, cycling 802.11b (DSSS/CCK) and 802.11g/n (OFDM) PHY rates.
//   * BLE:   non-connectable advertising at a fast interval on ch 37/38/39.
//
// Serial commands (115200 baud), one per line:
//   w <ch>   Wi-Fi only on channel <ch> (1-13)       b        BLE advertising only
//   m <ch>   Wi-Fi + BLE (coexistence, time-sliced)  d <ms>   mean gap between Wi-Fi frames
//   p <q>    max TX power in 0.25 dBm units (8..84)  s        stop everything
#include <WiFi.h>
#include <esp_wifi.h>
#include <BLEDevice.h>
#include <BLEAdvertising.h>

static int gChannel = 6;
static bool gWifi = false, gBle = false;
static int gGapMs = 4;
static BLEAdvertising* gAdv = nullptr;

static const wifi_phy_rate_t RATES[] = {
  WIFI_PHY_RATE_1M_L, WIFI_PHY_RATE_11M_L, WIFI_PHY_RATE_6M, WIFI_PHY_RATE_24M,
  WIFI_PHY_RATE_54M, WIFI_PHY_RATE_MCS3_LGI, WIFI_PHY_RATE_MCS7_LGI,
};

// Broadcast data frame from the AP's own BSSID (no impersonation of other networks).
static uint8_t frame[1500];
static size_t buildFrame(size_t payload) {
  uint8_t mac[6];
  esp_wifi_get_mac(WIFI_IF_AP, mac);
  const uint8_t hdr[] = {0x08, 0x02, 0x00, 0x00,              // data, FromDS
                         0xff, 0xff, 0xff, 0xff, 0xff, 0xff}; // addr1 broadcast
  memcpy(frame, hdr, sizeof(hdr));
  memcpy(frame + 10, mac, 6);   // addr2 = BSSID
  memcpy(frame + 16, mac, 6);   // addr3 = SA
  frame[22] = frame[23] = 0;
  for (size_t i = 0; i < payload; i++) frame[24 + i] = esp_random() & 0xff;
  return 24 + payload;
}

static void startWifi() {
  WiFi.mode(WIFI_AP);
  WiFi.softAP("CLASSIDYNE-TEST", "classidyne-bench", gChannel, 0, 1);
  esp_wifi_set_ps(WIFI_PS_NONE);
  gWifi = true;
}

static void stopWifi() {
  gWifi = false;
  WiFi.softAPdisconnect(true);
  WiFi.mode(WIFI_OFF);
}

static void startBle() {
  if (!gAdv) {
    BLEDevice::init("CLASSIDYNE-BLE");
    gAdv = BLEDevice::getAdvertising();
    BLEAdvertisementData d;
    d.setName("CLASSIDYNE-BLE");
    d.setManufacturerData(String("\xff\xff classidyne bench", 18));
    gAdv->setAdvertisementData(d);
    gAdv->setMinInterval(0x20);   // 20 ms
    gAdv->setMaxInterval(0x30);   // 30 ms
  }
  gAdv->start();
  gBle = true;
}

static void stopBle() {
  if (gAdv) gAdv->stop();
  gBle = false;
}

static void setPower(int q) {
  q = constrain(q, 8, 84);
  esp_wifi_set_max_tx_power(q);
  Serial.printf("tx power %.1f dBm\n", q / 4.0);
}

static void handle(String line) {
  line.trim();
  if (!line.length()) return;
  char c = line[0];
  int arg = line.length() > 2 ? line.substring(2).toInt() : 0;
  if (c == 'w' || c == 'm') {
    if (arg >= 1 && arg <= 13) gChannel = arg;
    stopBle();
    stopWifi();
    startWifi();
    if (c == 'm') startBle();
    setPower(32);
  } else if (c == 'b') {
    stopWifi();
    startBle();
  } else if (c == 'd') {
    gGapMs = max(1, arg);
  } else if (c == 'p') {
    setPower(arg);
  } else if (c == 's') {
    stopBle();
    stopWifi();
  }
  Serial.printf("mode wifi=%d ble=%d ch=%d gap=%dms\n", gWifi, gBle, gChannel, gGapMs);
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("# esp32_traffic ready: w <ch> | b | m <ch> | d <ms> | p <q> | s");
  handle("s");  // boot idle; the capture script enables a mode over serial
}

void loop() {
  if (Serial.available()) handle(Serial.readStringUntil('\n'));
  if (!gWifi) {
    delay(5);
    return;
  }
  // A burst of frames at one PHY rate, then switch, so waterfalls show both DSSS and OFDM shapes.
  wifi_phy_rate_t rate = RATES[esp_random() % (sizeof(RATES) / sizeof(RATES[0]))];
  esp_wifi_config_80211_tx_rate(WIFI_IF_AP, rate);
  int n = 5 + esp_random() % 40;
  for (int i = 0; i < n; i++) {
    size_t len = buildFrame(40 + esp_random() % 1400);
    esp_wifi_80211_tx(WIFI_IF_AP, frame, len, true);
    delay(gGapMs / 2 + esp_random() % (gGapMs + 1));
  }
  delay(esp_random() % 60);  // idle gaps between traffic bursts
}
