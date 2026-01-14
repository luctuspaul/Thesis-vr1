import cv2
import numpy as np

class YOLODetector:
    def __init__(self, config_path='yolov4-tiny.cfg', weights_path='yolov4-tiny.weights', 
                 names_path='coco.names', confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initialize YOLOv4 detector
        
        Args:
            config_path: Path to YOLOv4 config file (default: yolov4.cfg)
            weights_path: Path to YOLOv4 weights file (default: yolov4.weights)
            names_path: Path to class names file
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Load YOLO
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load class names
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')
    
    def detect(self, frame):
        """
        Detect objects in frame
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            detections: List of dictionaries containing detection info
            annotated_frame: Frame with bounding boxes drawn
        """
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), 
                                     swapRB=True, crop=False)
        
        # Set input and perform forward pass
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layers)
        
        # Initialize lists for detected bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        
        # Loop over each output layer
        for output in layer_outputs:
            # Loop over each detection
            for detection in output:
                # Extract class ID and confidence
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter out weak predictions
                if confidence > self.confidence_threshold:
                    # Scale bounding box coordinates back to image size
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")
                    
                    # Calculate top-left corner
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))
                    
                    # Update lists
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                                   self.confidence_threshold, 
                                   self.nms_threshold)
        
        # Prepare detection results
        detections = []
        annotated_frame = frame.copy()
        
        if len(indices) > 0:
            for i in indices.flatten():
                # Extract bounding box coordinates
                x, y, w, h = boxes[i]
                
                # Get class info
                class_id = class_ids[i]
                label = self.classes[class_id]
                confidence = confidences[i]
                color = [int(c) for c in self.colors[class_id]]
                
                # Store detection info
                detections.append({
                    'label': label,
                    'confidence': confidence,
                    'box': (x, y, w, h),
                    'class_id': class_id
                })
                
                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                text = f"{label}: {confidence:.2f}"
                cv2.putText(annotated_frame, text, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return detections, annotated_frame
    
    def detect_specific_classes(self, frame, target_classes=['car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle']):
        """
        Detect only specific classes of objects
        
        Args:
            frame: Input frame
            target_classes: List of class names to detect
            
        Returns:
            filtered_detections: List of detections for target classes only
            annotated_frame: Frame with bounding boxes for target classes
        """
        detections, annotated_frame = self.detect(frame)
        
        # Filter detections for target classes
        filtered_detections = [d for d in detections if d['label'] in target_classes]
        
        return filtered_detections, annotated_frame
    
    def count_objects(self, detections):
        """
        Count detected objects by class
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            counts: Dictionary with class names as keys and counts as values
        """
        counts = {}
        for detection in detections:
            label = detection['label']
            counts[label] = counts.get(label, 0) + 1
        return counts