import cv2
import numpy as np
import pandas as pd
import re
import pytesseract
import os
import sys

class Logger:
    LEVELS = {"DEBUG": 0, "INFO": 1, "ERROR": 2}

    def __init__(self, level="INFO"):
        self.level = self.LEVELS.get(level.upper(), 1)
        self.log_lines = []

    def debug(self, msg):
        if self.level <= 0:
            print(f"[DEBUG] {msg}")
        self.log_lines.append(f"[DEBUG] {msg}")

    def info(self, msg):
        if self.level <= 1:
            print(f"[INFO] {msg}")
        self.log_lines.append(f"[INFO] {msg}")

    def error(self, msg):
        if self.level <= 2:
            print(f"[ERROR] {msg}")
        self.log_lines.append(f"[ERROR] {msg}")

def load_config(filepath="config.txt"):
    config = {
        "TESSERACT_PATH": r"C:\tesseract.exe",
        "MIN_INDENT_SIZE_UM": 10.0,
        "MIN_INTERSECT_ANGLE_DEG": 30.0,
        "USE_ASTM": True,
        "USE_ISO": True,
        "DEFAULT_SCALE_UM_PX": 1.0,
        "DEBUG_LEVEL": "INFO"
    }
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    if key in ["MIN_INDENT_SIZE_UM", "MIN_INTERSECT_ANGLE_DEG", "DEFAULT_SCALE_UM_PX"]:
                        config[key] = float(val)
                    elif key in ["USE_ASTM", "USE_ISO"]:
                        config[key] = val.lower() == "true"
                    else:
                        config[key] = val
    return config

# Load configuration and initialize logger
config = load_config()
logger = Logger(level=config.get("DEBUG_LEVEL", "INFO"))

# Configure tesseract path from config
if os.path.exists(config["TESSERACT_PATH"]):
    pytesseract.pytesseract.tesseract_cmd = config["TESSERACT_PATH"]

def step1_image_data_loader(image_path):
    """STEP 1: IMAGE DATA LOADER"""
    print(f"[STEP 1] Loading image: {image_path}")
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

def step2_dynamic_scale_calibration(img, config):
    """STEP 2: DYNAMIC SCALE CALIBRATION"""
    print("[STEP 2] Calibrating scale...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Isolate scale bar region (usually bottom right)
    roi = gray[int(h*0.85):h, int(w*0.75):w]
    _, white_box_thresh = cv2.threshold(roi, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(white_box_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    physical_value = None
    pixel_width = None

    if contours:
        large_contour = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(large_contour)
        scale_box = roi[by:by+bh, bx:bx+bw]
        
        _, black_line_thresh = cv2.threshold(scale_box, 50, 255, cv2.THRESH_BINARY_INV)
        line_contours, _ = cv2.findContours(black_line_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pixel_width = next((float(cv2.boundingRect(lc)[2]) for lc in line_contours if cv2.boundingRect(lc)[2] > 50 and cv2.boundingRect(lc)[3] < 12), float(bw) * 0.9)

        try:
            text = pytesseract.image_to_string(scale_box, config='--psm 6')
            numeric_match = re.search(r'\d+', text)
            if numeric_match:
                physical_value = float(numeric_match.group())
        except Exception:
            pass

    if physical_value is None or pixel_width is None:
        print("[WARNING] Automatic scale calibration failed.")
        try:
            user_val = input("Please enter the scale in um/pixel (or leave blank for manual calibration if not visible): ")
            if user_val.strip():
                return float(user_val)
            else:
                user_um = input("Enter scale physical length (um) [or leave blank for default]: ")
                if not user_um.strip():
                    return config.get("DEFAULT_SCALE_UM_PX", 1.0)
                physical_um = float(user_um)
                pixel_len = float(input("Enter scale pixel length: "))
                return physical_um / pixel_len
        except ValueError:
            default_val = config.get("DEFAULT_SCALE_UM_PX", 1.0)
            print(f"[ERROR] Invalid input. Defaulting to {default_val} um/px.")
            return default_val

    c_factor = physical_value / pixel_width
    print(f"[INFO] Step 2 Calibration: {pixel_width} px = {physical_value} um ({c_factor:.4f} um/px)")
    return c_factor

def step3_edge_curve_linearity_regression(img):
    """STEP 3: EDGE CURVE LINEARITY REGRESSION (RSQ ENGINE)"""
    print("[STEP 3] Performing edge linearity regression...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark_zone_mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    edge_contours, _ = cv2.findContours(dark_zone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not edge_contours:
        print("[ERROR] No edge contour detected.")
        sys.exit(1)

    edge_contour = max(edge_contours, key=cv2.contourArea)
    points = edge_contour.reshape(-1, 2)
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)

    def calculate_rsq(x, y, poly):
        y_pred = poly(x)
        y_mean = np.mean(y)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y_mean)**2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Linear Fit (Degree 1)
    lin_coeffs = np.polyfit(x, y, 1)
    lin_poly = np.poly1d(lin_coeffs)
    lin_rsq = calculate_rsq(x, y, lin_poly)

    # Quadratic Fit (Degree 2)
    quad_coeffs = np.polyfit(x, y, 2)
    quad_poly = np.poly1d(quad_coeffs)
    quad_rsq = calculate_rsq(x, y, quad_poly)

    regression_results = {
        'linear': {'coeffs': lin_coeffs, 'rsq': lin_rsq},
        'quadratic': {'coeffs': quad_coeffs, 'rsq': quad_rsq}
    }
    
    print(f"[INFO] Linear RSQ: {lin_rsq:.4f}, Quadratic RSQ: {quad_rsq:.4f}")
    return points, regression_results

def step4_specimen_border_profiling_fitting(edge_points, regression_results):
    """STEP 4: SPECIMEN BORDER PROFILING & FITTING"""
    print("[STEP 4] Fitting specimen border...")
    lin_rsq = regression_results['linear']['rsq']
    quad_rsq = regression_results['quadratic']['rsq']

    # Choose best model (quadratic only if significantly better)
    if quad_rsq > lin_rsq + 0.01:
        print("[INFO] Selected Quadratic Model.")
        best_coeffs = regression_results['quadratic']['coeffs']
        model = np.poly1d(best_coeffs)
    else:
        print("[INFO] Selected Linear Model.")
        best_coeffs = regression_results['linear']['coeffs']
        model = np.poly1d(best_coeffs)
    
    return model

def step5_indentation_center_isolation_measurement(img, border_model, c_factor):
    """STEP 5: INDENTATION CENTER ISOLATION & DISTANCE MEASUREMENT"""
    print("[STEP 5] Isolating indents and measuring distances...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_indents = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    indent_contours, _ = cv2.findContours(thresh_indents, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    indent_data = []
    for c in indent_contours:
        if 200 < cv2.contourArea(c) < 6000:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                # Perpendicular distance calculation to the fitted border model
                # Approximate shortest distance by sampling the curve if it's quadratic
                # or using point-line distance if it's linear.
                # For simplicity, let's sample points on the model across image width.
                x_range = np.linspace(0, img.shape[1], img.shape[1])
                y_range = border_model(x_range)
                
                # Calculate Euclidean distance to all points on the curve
                dists = np.sqrt((x_range - cX)**2 + (y_range - cY)**2)
                pixel_dist = np.min(dists)
                micron_dist = pixel_dist * c_factor
                
                indent_data.append({
                    "contour": c,
                    "cX": cX, "cY": cY,
                    "dist_um": micron_dist,
                    "pixel_dist": pixel_dist
                })
                
    # Sort by distance from edge
    indent_data = sorted(indent_data, key=lambda k: k['dist_um'])
    return indent_data

def calculate_indent_geometry(contour, c_factor):
    """Analyzes the contour of an indentation to find its geometric properties."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        approx = np.intp(box).reshape(-1, 1, 2)

    pts = approx.reshape(4, 2)
    pts_sorted_y = pts[np.argsort(pts[:, 1]), :]
    top_pt, bottom_pt = pts_sorted_y[0], pts_sorted_y[3]
    remaining = pts_sorted_y[1:3, :]
    remaining_sorted_x = remaining[np.argsort(remaining[:, 0]), :]
    left_pt, right_pt = remaining_sorted_x[0], remaining_sorted_x[1]

    v1, v2 = top_pt - bottom_pt, left_pt - right_pt
    d1_um, d2_um = np.linalg.norm(v1) * c_factor, np.linalg.norm(v2) * c_factor
    angle1, angle2 = np.degrees(np.arctan2(v1[1], v1[0])), np.degrees(np.arctan2(v2[1], v2[0]))

    if angle1 > 90: angle1 -= 180
    elif angle1 < -90: angle1 += 180
    if angle2 > 90: angle2 -= 180
    elif angle2 < -90: angle2 += 180

    intersect_angle = abs(angle1 - angle2)
    if intersect_angle > 90: intersect_angle = 180 - intersect_angle

    return d1_um, d2_um, angle1, angle2, intersect_angle

def step6_vickers_geometric_diamond_diagnostics(indent_data, c_factor, config):
    """STEP 6: VICKERS GEOMETRIC DIAMOND DIAGNOSTICS"""
    print("[STEP 6] Running diamond diagnostics...")
    min_size = config.get("MIN_INDENT_SIZE_UM", 10.0)
    min_angle = config.get("MIN_INTERSECT_ANGLE_DEG", 30.0)

    for pt in indent_data:
        d1, d2, ang1, ang2, intersect = calculate_indent_geometry(pt['contour'], c_factor)
        pt.update({
            "d1_um": d1, "d2_um": d2, "avg_d_um": (d1 + d2) / 2.0,
            "ang1": ang1, "ang2": ang2, "intersect": intersect
        })

        # Quality Filters
        if not (d1 >= min_size and d2 >= min_size):
            pt["status"] = f"REJECT (<{min_size}um)"
        elif not (intersect > min_angle):
            pt["status"] = f"REJECT (Angle <={min_angle}deg)"
        else:
            pt["status"] = "ACCEPT"

    return indent_data

def step7_astm_iso_standard_compliance_validation(indent_data, config):
    """STEP 7: ASTM E384 / ISO 6507-1 STANDARD COMPLIANCE VALIDATION"""
    print("[STEP 7] Validating standard compliance...")
    use_astm = config.get("USE_ASTM", True)
    use_iso = config.get("USE_ISO", True)

    # Standard 2.5d rule for edge distance
    for pt in indent_data:
        if pt["status"] == "ACCEPT":
            if use_astm or use_iso:
                if pt["dist_um"] < 2.5 * pt["avg_d_um"]:
                    pt["status"] = "REJECT (Too close to edge)"
    return indent_data

def step8_output_asset_generation_log_appender(image_path, img, indent_data, log_lines, border_model):
    """STEP 8: OUTPUT ASSET GENERATION & LOG APPENDER"""
    print("[STEP 8] Generating outputs...")
    base_name, _ = os.path.splitext(image_path)
    csv_output = f"{base_name}_profile.csv"
    img_output = f"{base_name}_measured.png"
    txt_output = f"{base_name}_Measurement.txt"

    annotated_img = img.copy()
    h, w = img.shape[:2]
    
    # Draw border
    x_range = np.linspace(0, w, w).astype(int)
    y_range = border_model(x_range).astype(int)
    for i in range(len(x_range) - 1):
        if 0 <= y_range[i] < h and 0 <= y_range[i+1] < h:
            cv2.line(annotated_img, (x_range[i], y_range[i]), (x_range[i+1], y_range[i+1]), (0, 0, 255), 2)

    full_log_data = []
    qualified_report_data = []

    accept_count = 0
    for idx, pt in enumerate(indent_data):
        step_size = 0.0 if idx == 0 else pt['dist_um'] - indent_data[idx-1]['dist_um']
        
        full_log_data.append({
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
            accept_count += 1
            qualified_report_data.append({
                "Indentation_No": accept_count,
                "Averaged_Diagonal_Size_um": round(pt['avg_d_um'], 1),
                "Distance_From_Edge_um": round(pt['dist_um'], 1),
                "Step_Size_um": round(step_size, 1) if idx > 0 else "Baseline"
            })
            cv2.drawMarker(annotated_img, (pt['cX'], pt['cY']), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(annotated_img, str(accept_count), (pt['cX'] - 25, pt['cY'] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.drawMarker(annotated_img, (pt['cX'], pt['cY']), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 1)

    df_full_log = pd.DataFrame(full_log_data)
    df_final_report = pd.DataFrame(qualified_report_data)

    df_final_report.to_csv(csv_output, index=False)
    cv2.imwrite(img_output, annotated_img)
    
    log_lines.append(f"\n[SUCCESS] Extracted {len(qualified_report_data)} / {len(indent_data)} qualified indentation points.")
    log_lines.append(f"  -> Clean CSV Table saved to:  {csv_output}")
    log_lines.append(f"  -> Annotated Image saved to:  {img_output}")
    log_lines.append(f"  -> Master Log Text saved to:  {txt_output}\n")
    
    with open(txt_output, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_lines) + "\n")
        f.write("="*95 + "\n")
        f.write("TABLE 1: COMPLETE GEOMETRIC RUN LOG (INCLUDES REJECTS)\n")
        f.write("="*95 + "\n")
        f.write(df_full_log.to_string(index=False) + "\n\n")
        f.write("="*95 + "\n")
        f.write("TABLE 2: FINAL QUALIFIED ENGINEERING PROFILE REPORT\n")
        f.write("="*95 + "\n")
        if not df_final_report.empty:
            f.write(df_final_report.to_string(index=False) + "\n")
        else:
            f.write("NO QUALIFIED POINTS FOUND\n")

    print("="*95 + "\nFINAL OUTPUT DATA REPORT\n" + "="*95)
    if not df_final_report.empty:
        print(df_final_report.to_string(index=False))
    else:
        print("NO QUALIFIED POINTS FOUND")

def process_hardness_profile(image_path):
    logger.info(f"--- STARTING WORKFLOW for {os.path.basename(image_path)} ---")

    logger.info("STEP 1: IMAGE DATA LOADER")
    img = step1_image_data_loader(image_path)
    logger.debug(f"Step 1 Output: img.shape={img.shape if img is not None else 'None'}")

    logger.info("STEP 2: DYNAMIC SCALE CALIBRATION")
    c_factor = step2_dynamic_scale_calibration(img, config)
    logger.debug(f"Step 2 Output: c_factor={c_factor:.4f} um/px")

    logger.info("STEP 3: EDGE CURVE LINEARITY REGRESSION (RSQ ENGINE)")
    edge_points, regression_results = step3_edge_curve_linearity_regression(img)
    logger.debug(f"Step 3 Output: {len(edge_points)} edge points, Linear RSQ={regression_results['linear']['rsq']:.4f}, Quadratic RSQ={regression_results['quadratic']['rsq']:.4f}")

    logger.info("STEP 4: SPECIMEN BORDER PROFILING & FITTING")
    border_model = step4_specimen_border_profiling_fitting(edge_points, regression_results)
    logger.debug(f"Step 4 Output: border_model={border_model}")

    logger.info("STEP 5: INDENTATION CENTER ISOLATION & DISTANCE MEASUREMENT")
    indent_data = step5_indentation_center_isolation_measurement(img, border_model, c_factor)
    logger.debug(f"Step 5 Output: Found {len(indent_data)} indents")

    logger.info("STEP 6: VICKERS GEOMETRIC DIAMOND DIAGNOSTICS")
    indent_data = step6_vickers_geometric_diamond_diagnostics(indent_data, c_factor, config)
    logger.debug(f"Step 6 Output: Processed {len(indent_data)} indents")

    logger.info("STEP 7: ASTM E384 / ISO 6507-1 STANDARD COMPLIANCE VALIDATION")
    indent_data = step7_astm_iso_standard_compliance_validation(indent_data, config)
    logger.debug(f"Step 7 Output: Validated {len(indent_data)} indents")

    logger.info("STEP 8: OUTPUT ASSET GENERATION & LOG APPENDER")
    step8_output_asset_generation_log_appender(image_path, img, indent_data, logger.log_lines, border_model)
    logger.info("--- WORKFLOW COMPLETE ---")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Missing image path argument.")
        sys.exit(1)
        
    input_path = sys.argv[1]
    process_hardness_profile(input_path)
