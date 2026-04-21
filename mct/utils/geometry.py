"""
Geometry utility functions for MCT system.
"""


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    box format: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area


def is_face_inside_body(face_box, body_box):
    """
    Check if face center is inside the body bounding box.
    
    Args:
        face_box: [x1, y1, x2, y2]
        body_box: [x1, y1, x2, y2]
    
    Returns:
        bool: True if face center is inside body box
    """
    fx_center = (face_box[0] + face_box[2]) / 2
    fy_center = (face_box[1] + face_box[3]) / 2
    
    bx1, by1, bx2, by2 = body_box
    
    if bx1 < fx_center < bx2 and by1 < fy_center < by2:
        return True
    return False
