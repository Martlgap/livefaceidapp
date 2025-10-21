import gradio as gr
import numpy as np
import onnxruntime as rt
import mediapipe as mp
import os
import cv2
from typing import List
from skimage.transform import SimilarityTransform
from types import SimpleNamespace
from sklearn.metrics.pairwise import cosine_distances


# ---------------------------------------------------------------------------------------------------------------------

# Define a class to store a detection
class Detection(SimpleNamespace):
    bbox: List[List[float]] = None
    landmarks: List[List[float]] = None


# Define a class to store an identity
class Identity(SimpleNamespace):
    detection: Detection = Detection()
    name: str = None
    embedding: np.ndarray = None
    face: np.ndarray = None


# Define a class to store a match
class Match(SimpleNamespace):
    subject_id: Identity = Identity()
    gallery_id: Identity = Identity()
    distance: float = None
    name: str = None


# Similarity threshold for face matching
SIMILARITY_THRESHOLD = 1.0

# Init face detector and face recognizer
FACE_RECOGNIZER = rt.InferenceSession(
    "model.onnx", providers=rt.get_available_providers()
)
FACE_DETECTOR = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=7
)

# ---------------------------------------------------------------------------------------------------------------------


def detect_faces(frame: np.ndarray) -> List[Detection]:
    # Process the frame with the face detector
    result = FACE_DETECTOR.process(frame)

    # Initialize an empty list to store the detected faces
    detections = []

    # Check if any faces were detected
    if result.multi_face_landmarks:
        # Iterate over each detected face
        for count, detection in enumerate(result.multi_face_landmarks):
            # Select 5 Landmarks
            landmark_indices = [470, 475, 1, 57, 287]
            five_landmarks = np.asarray(detection.landmark)[landmark_indices]

            # Extract the x and y coordinates of the landmarks of interest
            landmarks = np.asarray([
                [landmark.x * frame.shape[1], landmark.y * frame.shape[0]]
                for landmark in five_landmarks
            ])

            # Extract the x and y coordinates of all landmarks
            all_x_coords = [
                landmark.x * frame.shape[1] for landmark in detection.landmark
            ]
            all_y_coords = [
                landmark.y * frame.shape[0] for landmark in detection.landmark
            ]

            # Compute the bounding box of the face
            x_min, x_max = int(min(all_x_coords)), int(max(all_x_coords))
            y_min, y_max = int(min(all_y_coords)), int(max(all_y_coords))
            bbox = [[x_min, y_min], [x_max, y_max]]

            # Create a Detection object for the face
            detection = Detection(
                idx=count, bbox=bbox, landmarks=landmarks, confidence=None
            )

            # Add the detection to the list
            detections.append(detection)

    # Return the list of detections
    return detections


def recognize_faces(
    frame: np.ndarray, detections: List[Detection]
) -> List[Identity]:
    if not detections:
        return []

    identities = []
    for detection in detections:
        # ALIGNMENT -------------------------------------------------------
        # Target landmark coordinates (as used in training)
        landmarks_target = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
        
        tform = SimilarityTransform()
        tform.estimate(detection.landmarks, landmarks_target)
        tmatrix = tform.params[0:2, :]
        face_aligned = cv2.warpAffine(
            frame, tmatrix, (112, 112), borderValue=0.0
        )
        # -----------------------------------------------------------------

        # INFERENCE -------------------------------------------------------
        # Inference face embeddings with onnxruntime
        input_image = (
            np.asarray([face_aligned]).astype(np.float32) / 255.0
        ).clip(0.0, 1.0)
        embedding = FACE_RECOGNIZER.run(
            None, {"input_image": input_image}
        )[0][0]
        # -----------------------------------------------------------------

        # Create Identity object
        identities.append(Identity(
            detection=detection, embedding=embedding, face=face_aligned
        ))

    return identities


def match_faces(
    subjects: List[Identity], gallery: List[Identity]
) -> List[Match]:
    if len(gallery) == 0 or len(subjects) == 0:
        return []

    # Get Embeddings
    embs_gal = np.asarray([identity.embedding for identity in gallery])
    embs_det = np.asarray([identity.embedding for identity in subjects])

    # Calculate Cosine Distances
    cos_distances = cosine_distances(embs_det, embs_gal)

    # Find Matches
    matches = []
    for ident_idx, identity in enumerate(subjects):
        dists_to_identity = cos_distances[ident_idx]
        idx_min = np.argmin(dists_to_identity)
        if dists_to_identity[idx_min] < SIMILARITY_THRESHOLD:
            matches.append(Match(
                subject_id=identity,
                gallery_id=gallery[idx_min],
                distance=dists_to_identity[idx_min]
            ))

    # Sort Matches by identity_idx
    matches = sorted(matches, key=lambda match: match.gallery_id.name)

    return matches


def draw_annotations(
    frame: np.ndarray, detections: List[Detection], matches: List[Match]
) -> np.ndarray:
    shape = np.asarray(frame.shape[:2][::-1])

    # Upscale frame to 1080p for better visualization
    frame = cv2.resize(frame, (1920, 1080))
    upscale_factor = np.asarray([1920 / shape[0], 1080 / shape[1]])
    shape = np.asarray(frame.shape[:2][::-1])

    # Make frame writeable (for better performance)
    frame.flags.writeable = True

    # Draw Detections
    for detection in detections:
        # Draw Landmarks
        for landmark in detection.landmarks:
            cv2.circle(
                frame,
                (landmark * upscale_factor).astype(int),
                2,
                (255, 255, 255),
                -1
            )

        # Draw Bounding Box
        cv2.rectangle(
            frame,
            (detection.bbox[0] * upscale_factor).astype(int),
            (detection.bbox[1] * upscale_factor).astype(int),
            (255, 0, 0),
            2,
        )

        # Draw Index
        cv2.putText(
            frame,
            str(detection.idx),
            (
                ((detection.bbox[1][0] + 2) * upscale_factor[0]).astype(int),
                ((detection.bbox[1][1] + 2) * upscale_factor[1]).astype(int),
            ),
            cv2.LINE_AA,
            0.5,
            (0, 0, 0),
            2,
        )

    # Draw Matches
    for match in matches:
        detection = match.subject_id.detection
        name = match.gallery_id.name

        # Draw Bounding Box in green
        cv2.rectangle(
            frame,
            (detection.bbox[0] * upscale_factor).astype(int),
            (detection.bbox[1] * upscale_factor).astype(int),
            (0, 255, 0),
            2,
        )

        # Draw Banner
        banner_y1 = (
            detection.bbox[0][1] * upscale_factor[1] - (shape[1] // 25)
        ).astype(int)
        banner_y2 = (detection.bbox[0][1] * upscale_factor[1]).astype(int)
        
        cv2.rectangle(
            frame,
            (
                (detection.bbox[0][0] * upscale_factor[0]).astype(int),
                banner_y1,
            ),
            (
                (detection.bbox[1][0] * upscale_factor[0]).astype(int),
                banner_y2,
            ),
            (255, 255, 255),
            -1,
        )

        # Draw Name
        name_x = (
            (detection.bbox[0][0] + shape[0] // 400) * upscale_factor[0]
        ).astype(int)
        name_y = (
            (detection.bbox[0][1] - shape[1] // 50) * upscale_factor[1]
        ).astype(int)
        
        cv2.putText(
            frame,
            name,
            (name_x, name_y),
            cv2.LINE_AA,
            0.7,
            (0, 0, 0),
            2,
        )

        # Draw Distance
        dist_x = (
            (detection.bbox[0][0] + shape[0] // 400) * upscale_factor[0]
        ).astype(int)
        dist_y = (
            (detection.bbox[0][1] - shape[1] // 350) * upscale_factor[1]
        ).astype(int)
        
        cv2.putText(
            frame,
            f" Distance: {match.distance:.2f}",
            (dist_x, dist_y),
            cv2.LINE_AA,
            0.5,
            (0, 0, 0),
            2,
        )

    return frame


def process_webcam_frame(frame: np.ndarray, gallery: List[Identity]) -> np.ndarray:
    """Process a single webcam frame for face recognition"""
    if frame is None:
        return frame
    
    # Run face detection
    detections = detect_faces(frame)

    # Run face recognition
    subjects = recognize_faces(frame, detections)

    # Run face matching
    matches = match_faces(subjects, gallery)

    # Draw annotations
    annotated_frame = draw_annotations(frame, detections, matches)

    return annotated_frame


# ---------------------------------------------------------------------------------------------------------------------

# Global gallery to store uploaded faces
global_gallery = []


def add_face_to_gallery(image_files):
    """Process uploaded images and add faces to gallery"""
    global global_gallery
    global_gallery = []  # Reset gallery
    
    gallery_previews = []
    gallery_names = []
    
    if image_files is None:
        return [], "No images uploaded"
    
    # Handle single file or list of files
    if not isinstance(image_files, list):
        image_files = [image_files]
    
    for idx, image_file in enumerate(image_files):
        if image_file is None:
            continue
            
        # Read image from file path
        img = cv2.imread(image_file)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = os.path.splitext(os.path.basename(image_file))[0]

        # Detect faces
        detections = detect_faces(img)

        if detections:
            # Recognize faces (take only the first face)
            subjects = recognize_faces(img, detections[:1])
            
            if subjects:
                # Add to global gallery
                identity = Identity(
                    name=filename,
                    embedding=subjects[0].embedding,
                    face=subjects[0].face
                )
                global_gallery.append(identity)
                
                # Add to preview
                gallery_previews.append(subjects[0].face)
                gallery_names.append(filename)
    
    status_msg = f"Added {len(global_gallery)} face(s) to gallery"
    return gallery_previews, status_msg


def process_webcam(frame):
    """Process webcam frame with current gallery"""
    if frame is None:
        return None, None, None, ""
    
    # Run face detection
    detections = detect_faces(frame)

    # Run face recognition
    subjects = recognize_faces(frame, detections)

    # Run face matching
    matches = match_faces(subjects, global_gallery)

    # Draw annotations
    annotated_frame = draw_annotations(frame, detections, matches)
    
    # Prepare match information for display
    match_info = []
    matched_gallery_faces = []
    scores_text = ""
    
    if matches:
        scores_text = "**🎯 Live Match Results:**\n\n"
        scores_text += f"**Total Matches Found:** {len(matches)}\n\n"
        
        for i, match in enumerate(matches, 1):
            match_info.append({
                "name": match.gallery_id.name,
                "distance": match.distance,
                "face": match.gallery_id.face
            })
            matched_gallery_faces.append(match.gallery_id.face)
            
            # Calculate confidence percentage (inverse of distance)
            confidence = max(0, (1 - match.distance) * 100)
            
            scores_text += (
                f"**{i}. {match.gallery_id.name}**\n"
                f"   • Distance: {match.distance:.3f}\n"
                f"   • Confidence: {confidence:.1f}%\n\n"
            )
    else:
        scores_text = "**🔍 Scanning for faces...**\n\n"
        if detections:
            scores_text += (
                f"Detected {len(detections)} face(s) but no matches found.\n"
            )
            scores_text += "Try adjusting the similarity threshold."
        else:
            scores_text += "No faces detected in the current frame."
    
    return annotated_frame, matched_gallery_faces, scores_text


def update_threshold(threshold):
    """Update similarity threshold"""
    global SIMILARITY_THRESHOLD
    SIMILARITY_THRESHOLD = threshold
    return f"Similarity threshold updated to {threshold}"


# Create Gradio interface
with gr.Blocks(
    title="Live Webcam Face Recognition",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown("# 🎭 Live Webcam Face Recognition")
    gr.Markdown(
        "Upload reference images to build a face gallery, "
        "then capture images for face recognition."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📸 Face Gallery")
            
            # File upload for gallery images
            gallery_upload = gr.File(
                label="Upload Reference Images",
                file_count="multiple",
                file_types=["image"],
                type="filepath"
            )
            
            # Gallery preview
            gallery_preview = gr.Gallery(
                label="Current Gallery",
                show_label=True,
                columns=3,
                rows=2
            )
            
            gallery_status = gr.Textbox(
                label="Status",
                value="Upload images to build face gallery",
                interactive=False
            )
            
            # Similarity threshold control
            similarity_threshold = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=SIMILARITY_THRESHOLD,
                step=0.1,
                label="Similarity Threshold",
                info="Lower values = stricter matching"
            )
        
        with gr.Column(scale=2):
            gr.Markdown("## 📹 Live Face Recognition")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Webcam input and output
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Webcam Input"
                    )
                    webcam_output = gr.Image(
                        streaming=True,
                        label="Live Recognition Output"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Live Match Information")
                    
                    # Live scores display
                    live_scores = gr.Markdown(
                        value=(
                            "**🔍 Scanning for faces...**\n\n"
                            "Position your face in front of the camera."
                        ),
                        label="Match Scores"
                    )
                    
                    # Matched gallery faces display
                    matched_faces = gr.Gallery(
                        label="🖼️ Currently Matched Gallery Faces",
                        show_label=True,
                        columns=2,
                        rows=2,
                        height=300
                    )
                    
                    # Additional info
                    gr.Markdown(
                        "**💡 Tips:**\n"
                        "• Lower distance = better match\n"
                        "• Higher confidence = more certain\n"
                        "• Adjust threshold for sensitivity"
                    )
            
            # Set up streaming
            webcam_input.stream(
                fn=process_webcam,
                inputs=[webcam_input],
                outputs=[webcam_output, matched_faces, live_scores],
                stream_every=0.1
            )
    
    # Event handlers
    gallery_upload.change(
        fn=add_face_to_gallery,
        inputs=[gallery_upload],
        outputs=[gallery_preview, gallery_status]
    )
    
    similarity_threshold.change(
        fn=update_threshold,
        inputs=[similarity_threshold],
        outputs=[gallery_status]
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        show_error=True
    )
