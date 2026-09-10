import sys
import os
import json
import cv2
import numpy as np
from PIL import Image

def process_artwork(art_id, source_img_path, title, category):
    os.makedirs("artworks", exist_ok=True)
    img = cv2.imread(source_img_path)
    h, w, _ = img.shape

    # 1. Save original preview
    preview_path = f"artworks/{art_id}_preview.png"
    cv2.imwrite(preview_path, img)

    # 2. Extract clean transparent black lines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    
    lines_bgra = np.zeros((h, w, 4), dtype=np.uint8)
    lines_bgra[binary_mask > 0] = [20, 20, 20, 255]
    lines_path = f"artworks/{art_id}_lines.png"
    cv2.imwrite(lines_path, lines_bgra)

    # 3. Segment into solid color regions (Watershed / Contours)
    # Remove lines from color areas
    kernel = np.ones((3, 3), np.uint8)
    dilated_lines = cv2.dilate(binary_mask, kernel, iterations=1)
    color_areas = cv2.bitwise_not(dilated_lines)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(color_areas)

    # Extract dominant palette colors
    pil_img = Image.open(source_img_path).convert("RGB").resize((100, 100))
    raw_colors = sorted(pil_img.getcolors(maxcolors=10000), key=lambda x: x[0], reverse=True)
    palette = []
    def color_dist(c1, c2):
        return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2

    chosen_rgb = []
    for _, col in raw_colors:
        if col[0] > 235 and col[1] > 235 and col[2] > 235: continue
        if col[0] < 40 and col[1] < 40 and col[2] < 40: continue
        # Enforce distinct colors (prevent 7 variations of the same orange)
        if any(color_dist(col, existing) < 1600 for existing in chosen_rgb):
            continue
        chosen_rgb.append(col)
        palette.append("#{0:02X}{1:02X}{2:02X}".format(col[0], col[1], col[2]))
        if len(palette) >= 6:
            break

    # Build regions metadata
    regions = []
    reg_idx = 1
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        # Ignore canvas borders and speckles
        if 400 < area < (h * w * 0.7):
            cX = int(centroids[label_id][0])
            cY = int(centroids[label_id][1])

            # Sample color from original image
            sample_bgr = img[min(cY, h-1), min(cX, w-1)]
            sample_rgb = (int(sample_bgr[2]), int(sample_bgr[1]), int(sample_bgr[0]))

            best_palette_idx = 1
            min_dist = float('inf')
            for p_i, hex_c in enumerate(palette):
                pr = int(hex_c[1:3], 16)
                pg = int(hex_c[3:5], 16)
                pb = int(hex_c[5:7], 16)
                dist = (sample_rgb[0] - pr)**2 + (sample_rgb[1] - pg)**2 + (sample_rgb[2] - pb)**2
                if dist < min_dist:
                    min_dist = dist
                    best_palette_idx = p_i + 1

            # Extract region boundary polygon
            region_mask = (labels == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                poly = cv2.approxPolyDP(contours[0], 0.003 * cv2.arcLength(contours[0], True), True)
                points = [[int(pt[0][0]), int(pt[0][1])] for pt in poly]

                regions.append({
                    "id": f"reg_{reg_idx}",
                    "colorIndex": best_palette_idx,
                    "points": points,
                    "label": [cX, cY]
                })
                reg_idx += 1

    regions_path = f"artworks/{art_id}_regions.json"
    with open(regions_path, "w") as f:
        json.dump(regions, f, indent=2)

    # 4. Update catalog.json
    catalog_path = "catalog.json"
    catalog = {"artworks": []}
    if os.path.exists(catalog_path):
        try:
            with open(catalog_path, "r") as f:
                catalog = json.load(f)
        except Exception:
            pass

    github_base = "https://raw.githubusercontent.com/Leone-Studio/color-assets-cdn/main/artworks"
    entry = {
        "id": art_id,
        "title": title,
        "category": category,
        "difficulty": f"{len(regions)} Parts",
        "preview": f"{github_base}/{art_id}_preview.png",
        "lineArtUrl": f"{github_base}/{art_id}_lines.png",
        "regionsDataUrl": f"{github_base}/{art_id}_regions.json",
        "palette": palette,
        "totalRegions": len(regions)
    }

    catalog["artworks"] = [a for a in catalog.get("artworks", []) if a["id"] != art_id]
    catalog["artworks"].insert(0, entry)

    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"✨ Successfully processed {title} with {len(regions)} aligned regions and palette: {palette}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python process_clean_art.py <art_id> <path_to_source_img> <title> <category>")
    else:
        process_artwork(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
