import CoreML
import Vision

#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit

public typealias UIImage = NSImage

public extension NSImage {
    var cgImage: CGImage? {
        return self.cgImage(forProposedRect: nil, context: nil, hints: nil)
    }
}
#endif

public struct EasyCoreML {
    let model: MLModel
    
    public init(_ model: MLModel) {
        self.model = model
    }
    
    public func test(_ cgImage: CGImage, identifier: String, min: Double = 50) async throws -> Bool {
        guard let model = try? VNCoreMLModel(for: model) else {
            return false
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            let request = VNCoreMLRequest(model: model) {request, error in
                let results = request.results as? [VNClassificationObservation]
                
                guard let res = results?.first(where: {
                    $0.identifier == identifier
                }) else {
                    continuation.resume(returning: false)
                    return
                }
                
                continuation.resume(returning: Double(res.confidence) * Double(100) >= min)
            }
            
            let handler = VNImageRequestHandler(cgImage: cgImage)
            
            DispatchQueue.global(qos: .userInteractive).async {
                do {
                    try handler.perform([request])
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    public func test(_ data: Data, identifier: String, min: Double = 50) async throws -> Bool {
        guard let image = UIImage(data: data)?.cgImage else {
            throw NSError()
        }
        
        return try await self.test(image, identifier: identifier, min: min)
    }
    
#if canImport(UIKit) || canImport(AppKit)
    public func test(_ image: UIImage, identifier: String, min: Double = 50) async throws -> Bool {
        guard let image = image.cgImage else {
            throw NSError()
        }
        
        return try await self.test(image, identifier: identifier, min: min)
    }
#endif
}
