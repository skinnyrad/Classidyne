import CoreGraphics
import Foundation
// Prints "<windowID> <x> <y> <w> <h>" for the largest on-screen window owned by the given process name.
let owner = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "SDR++"
let list = CGWindowListCopyWindowInfo([.optionAll], kCGNullWindowID) as! [[String: Any]]
var best: (Int, CGRect)? = nil
for w in list where (w[kCGWindowOwnerName as String] as? String) == owner {
    let id = w[kCGWindowNumber as String] as! Int
    let b = CGRect(dictionaryRepresentation: w[kCGWindowBounds as String] as! CFDictionary)!
    if best == nil || b.width * b.height > best!.1.width * best!.1.height { best = (id, b) }
}
if let (id, b) = best { print(id, Int(b.minX), Int(b.minY), Int(b.width), Int(b.height)) } else { exit(1) }
