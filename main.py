import cv2
import numpy as np
import pandas as pd
import re
import pytesseract
import os
import sys

# Hardcode the local engine pointer before executing any OCR tasks
pytesseract.pytesseract.tesseract_cmd = r'C:\tesseract.exe'

def load_universal_image(image_path):
    """
    Robust image loader that automatically handles file format validation
    for JPG, PNG, and complex multi-page or high-bit depth TIFF files.
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] File does not exist: {image_path}")
        sys.exit(1)
        
    ext = os.path.splitext(image_path)[1].lower()
    print(f"[INFO] Detecting format: {ext.upper()} file profile.")

    # Primary attempt: Standard OpenCV loading matrix
    img = cv2.imread(image_path)
    
    # Fallback logic specifically for robust TIFF parsing if imread returns None
    if img is None and ext in ['.tif', '.tiff']:
        print("[INFO] Standard read failed. Attempting multi-page TIFF extraction handler...")
        ret, img_list = cv2.imreadmulti(image_path)
        if ret and len(img_list) > 0:
            img = img_list[0]  # Extract the primary image canvas layer
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            print("[ERROR] Multi-page TIFF decoder returned an empty frame array.")
            sys.exit(1)
            
    if img is None:
        print(f"[ERROR] Failed to decode image file matrix for: {image_path}.")
        sys.exit(1)
        
    return img

def calculate_indent_geometry(contour, c_factor):
    """
    Analyzes the contour of an indentation to find its 4 vertices,
    calculates the lengths of its two diagonals in microns, their absolute angles,
    and their intersecting angle.
    """
    # Approximate the contour to a polygon to find clear corners
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Fallback if approximation doesn't yield a 4-sided polygon clean
    if len(approx) != 4:
        # Use a minimum bounding rotated rectangle box as standard geometric fallback
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        approx = np.intp(box).reshape(-1, 1, 2)

    pts = approx.reshape(4, 2)
    
    # Sort vertices by their Y-coordinates to separate top/bottom from left/right
    pts_sorted_y = pts[np.argsort(pts[:, 1]), :]
    top_pt = pts_sorted_y[0]
    bottom_pt = pts_sorted_y[3]
    
    # The remaining two are left and right; sort them by X-coordinate
    remaining = pts_sorted_y[1:3, :]
    remaining_sorted_x = remaining[np.argsort(remaining[:, 0]), :]
    left_pt = remaining_sorted_x[0]
    right_pt = remaining_sorted_x[1]
    
    # Calculate diagonal pixel vectors
    v1 = top_pt - bottom_pt
    v2 = left_pt - right_pt
    
    # Compute physical sizes in microns
    d1_um = np.linalg.norm(v1) * c_factor
    d2_um = np.linalg.norm(v2) * c_factor
    
    # Calculate orientation angles relative to horizontal axis (in degrees)
    angle1 = np.degrees(np.arctan2(v1[1], v1[0]))
    angle2 = np.degrees(np.arctan2(v2[1], v2[0]))
    
    # Normalize angles to standard quadrant visualization [-90, 90]
    if angle1 > 90: angle1 -= 180
    elif angle1 < -90: angle1 += 180
    if angle2 > 90: angle2 -= 180
    elif angle2 < -90: angle2 += 180
        
    # Calculate the intersection angle between the two diagonal lines
    intersect_angle = abs(angle1 - angle2)
    if intersect_angle > 90:
        intersect_angle = 180 - intersect_angle
        
    return d1_um, d2_um, angle1, angle2, intersect_angle

def process_hardness_profile(image_path):
    # Call the robust file-type agnostic loader
    img = load_universal_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    log_lines = []
    def log_and_print(text_line):
        print(text_line)
        log_lines.append(text_line)

    log_and_print(f"[STARTING] Processing file: {os.path.basename(image_path)}")
    
    # -------------------------------------------------------------------------
    # STEP 1: DYNAMIC SCALE CALIBRATION WITH SAFE OCR WRAPPER
    # -------------------------------------------------------------------------
    roi = gray[int(h*0.85):h, int(w*0.75):w]
    _, white_box_thresh = cv2.threshold(roi, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(white_box_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("[ERROR] Could not isolate the white scale bar box container.")
        sys.exit(1)
        
    large_contour = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(large_contour)
    scale_box = roi[by:by+bh, bx:bx+bw]
    
    _, black_line_thresh = cv2.threshold(scale_box, 50, 255, cv2.THRESH_BINARY_INV)
    line_contours, _ = cv2.findContours(black_line_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pixel_width = next((float(cv2.boundingRect(lc)[2]) for lc in line_contours if cv2.boundingRect(lc)[2] > 50 and cv2.boundingRect(lc)[3] < 12), float(bw) * 0.9)
    
    try:
        text = pytesseract.image_to_string(scale_box, config='--psm 6')
        numeric_match = re.search(r'\d+', text)
        physical_value = float(numeric_match.group()) if numeric_match else (200.0 if pixel_width < 450 else 500.0)
    except Exception:
        physical_value = 200.0 if pixel_width < 450 else 500.0
        log_and_print("[INFO] OCR call bypassed or unavailable. Applied geometric scale fallback rule.")

    c_factor = physical_value / pixel_width
    log_and_print(f"[INFO] Step 1 Calibration: {pixel_width} px = {physical_value} um ({c_factor:.4f} um/px)")

    # -------------------------------------------------------------------------
    # STEP 2: EDGE LINE DETECTION (AX + BY + C = 0)
    # -------------------------------------------------------------------------
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_zone_mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    edge_contours, _ = cv2.findContours(dark_zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_contour = max(edge_contours, key=cv2.contourArea)
    
    vx, vy, x0, y0 = cv2.fitLine(edge_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    
    vx_val, vy_val = vx.item(), vy.item()
    x0_val, y0_val = x0.item(), y0.item()
    
    A, B, C = vy_val, -vx_val, vx_val * y0_val - vy_val * x0_val
    log_and_print(f"[INFO] Step 2 Edge Baseline: {A:.4f}*X + {B:.4f}*Y + {C:.2f} = 0")

    # -------------------------------------------------------------------------
    # STEP 3: INDENT DETECTION & RAW MEASUREMENTS
    # -------------------------------------------------------------------------
    _, thresh_indents = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    indent_contours, _ = cv2.findContours(thresh_indents, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_points = []
    for c in indent_contours:
        if 200 < cv2.contourArea(c) < 6000:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Perpendicular distance calculation to the fitted baseline
                pixel_dist = abs(A * cX + B * cY + C) / np.sqrt(A**2 + B**2)
                micron_dist = pixel_dist * c_factor
                
                # Run the advanced geometric diagnostics module
                d1, d2, ang1, ang2, intersect = calculate_indent_geometry(c, c_factor)
                
                # Integrated Quality Filters: Length >= 10um AND Intersection Angle > 30 deg
                if not (d1 >= 10.0 and d2 >= 10.0):
                    status = "REJECT (<10um)"
                elif not (intersect > 30.0):
                    status = "REJECT (Angle <=30deg)"
                else:
                    status = "ACCEPT"
                
                raw_points.append({
                    "cX": cX, "cY": cY, "dist_um": micron_dist,
                    "d1_um": d1, "d2_um": d2, "avg_d_um": (d1 + d2) / 2.0,
                    "ang1": ang1, "ang2": ang2, "intersect": intersect, "status": status
                })
                
    # Sort everything raw by distance from edge first (ascending)
    sorted_raw_points = sorted(raw_points, key=lambda k: k['dist_um'])

    # -------------------------------------------------------------------------
    # STEP 4: SEPARATE COMPILATION FOR RAW LOG vs QUALIFIED METRICS
    # -------------------------------------------------------------------------
    full_processing_log = []
    qualified_points = []
    annotated_img = img.copy()
    
    # Render the structural edge baseline boundary line (Red, thickness=2)
    if abs(B) > 0.001:
        cv2.line(annotated_img, (0, int(-C/B)), (w, int((-A*w - C)/B)), (0, 0, 255), 2)
    else:
        cv2.line(annotated_img, (int(-C/A), 0), (int(-C/A), h), (0, 0, 255), 2)

    # 1. Generate Full Processing Table Data Structure
    for idx, pt in enumerate(sorted_raw_points):
        step_size = 0.0 if idx == 0 else pt['dist_um'] - sorted_raw_points[idx-1]['dist_um']
        
        full_processing_log.append({
            "Raw_No": idx + 1,
            "Distance_From_Edge_um": round(pt['dist_um'], 1),
            "Step_Size_um": round(step_size, 1) if idx > 0 else "Baseline",
            "Diagonal_1_um": round(pt['d1_um'], 1),
            "Diagonal_2_um": round(pt['d2_um'], 1),
            "Diag_1_Angle": f"{pt['ang1']:.1f}°",
            "Diag_2_Angle": f"{pt['ang2']:.1f}°",
            "Intersect_Angle": f"{pt['intersect']:.1f}°",
            "Status": pt['status']
        })
        
        if pt['status'] == "ACCEPT":
            qualified_points.append(pt)

    # 2. Generate Qualified Clean Report Table Structure & Visual Markings
    final_report_data = []
    for idx, pt in enumerate(qualified_points):
        step_size = 0.0 if idx == 0 else pt['dist_um'] - qualified_points[idx-1]['dist_um']
        
        final_report_data.append({
            "Indentation_No": idx + 1,
            "Averaged_Diagonal_Size_um": round(pt['avg_d_um'], 1),
            "Distance_From_Edge_um": round(pt['dist_um'], 1),
            "Step_Size_um": round(step_size, 1) if idx > 0 else "Baseline"
        })
        
        # Only overlay markings on image if point is fully qualified
        cv2.drawMarker(annotated_img, (pt['cX'], pt['cY']), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
        cv2.putText(annotated_img, str(idx + 1), (pt['cX'] - 25, pt['cY'] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert datasets to pandas DataFrames
    df_full_log = pd.DataFrame(full_processing_log)
    df_final_report = pd.DataFrame(final_report_data)

    # -------------------------------------------------------------------------
    # STEP 5: FILE EXPORT & DUAL-TABLE STRUCTURING
    # -------------------------------------------------------------------------
    base_name, _ = os.path.splitext(image_path)
    csv_output = f"{base_name}_profile.csv"
    img_output = f"{base_name}_measured.png"
    txt_output = f"{base_name}_Measurement.txt"
    
    # Save clean qualified file directly to .csv and image
    df_final_report.to_csv(csv_output, index=False)
    cv2.imwrite(img_output, annotated_img)
    
    # Assemble Dual Table contents inside the plaintext measurement logger file
    log_and_print(f"\n[SUCCESS] Extracted {len(qualified_points)} / {len(sorted_raw_points)} qualified indentation points.")
    log_and_print(f"  -> Clean CSV Table saved to:  {csv_output}")
    log_and_print(f"  -> Annotated Image saved to:  {img_output}")
    log_and_print(f"  -> Master Log Text saved to:  {txt_output}\n")
    
    # Write both tables to the .txt file sequentially
    with open(txt_output, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_lines) + "\n")
        f.write("="*95 + "\n")
        f.write("TABLE 1: COMPLETE GEOMETRIC RUN LOG (INCLUDES REJECTS)\n")
        f.write("="*95 + "\n")
        f.write(df_full_log.to_string(index=False) + "\n\n")
        f.write("="*95 + "\n")
        f.write("TABLE 2: FINAL QUALIFIED ENGINEERING PROFILE REPORT\n")
        f.write("="*95 + "\n")
        f.write(df_final_report.to_string(index=False) + "\n")

    # Display the final summary on the screen console panel
    print("="*95 + "\nFINAL OUTPUT DATA REPORT\n" + "="*95)
    print(df_final_report.to_string(index=False))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Missing image path argument.")
        print("Usage in PowerShell: python profile_analyzer.py 'C:\\path\\to\\image.tif'")
        sys.exit(1)
        
    input_path = sys.argv[1]
    process_hardness_profile(input_path)
