import sys
import os
import json
import subprocess
import requests
import cv2
import numpy as np
from PIL import Image

def generate_and_process(art_id, title, category, prompt_subject):
    print(f"\n🎨 [1/5] Calling Higgsfield CLI for: {title}...")
    full_prompt = f"Cute chibi {prompt_subject}, bold thick black vector outlines, flat cell shading, clean sticker art, solid color fills, zero gradients, pure solid white background"
    
    cmd = [
        "higgsfield", "generate", "create", "recraft_v4_1",
        "--prompt", full_prompt,
        "--aspect_ratio", "1:1",
        "--wait"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    out_lines = [line.strip() for line in res.stdout.splitlines() if line.strip().startswith("http")]
    
    if not out_lines:
        print("❌ Higgsfield CLI failed to return image URL:")
        print(res.stderr or res.stdout)
        return
        
    image_url = out_lines[-1]
    print(f"✅ Image generated: {image_url}")

    os.makedirs("artworks", exist_ok=True)
    raw_path = os.path.join("artworks", f"{art_id}_preview.png")
    lines_path = os.path.join("artworks", f"{art_id}_lines.png")
    regions_path = os.path.join("artworks", f"{art_id}_regions.json")

    print(f"📥 [2/5] Downloading asset to {raw_path}...")
    img_data = requests.get(image_url).content
    with open(raw_path, "wb") as f:
        f.write(img_data)

    print("🧩 [3/5] Extracting Palette & Vector Contours...")
    img = cv2.imread(raw_path)
    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 140)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    line_art = 255 - dilated
    cv2.imwrite(lines_path, line_art)

    pil_img = Image.open(raw_path).convert("RGB").resize((100, 100))
    colors = sorted(pil_img.getcolors(maxcolors=10000), key=lambda x: x[0], reverse=True)
    palette = []
    for _, col in colors:
        if col[0] > 235 and col[1] > 235 and col[2] > 235:
            continue
        if col[0] < 35 and col[1] < 35 and col[2] < 35:
            continue
        hex_val = "#{:02X}{:02X}{:02X}".format(col[0], col[1], col[2])
        if hex_val not in palette:
            palette.append(hex_val)
        if len(palette) >= 5:
            break

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    color_idx = 1

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if 400 < area < (h * w * 0.65):
            epsilon = 0.005 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            points = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = points[0][0], points[0][1]

            regions.append({
                "id": f"reg_{i}",
                "colorIndex": color_idx,
                "points": points,
                "label": [cX, cY]
            })
            color_idx = (color_idx % len(palette)) + 1

    with open(regions_path, "w") as f:
        json.dump(regions, f, indent=2)

    print(f"📝 [4/5] Updating catalog.json with {len(regions)} regions...")
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

    print(f"🚀 [5/5] Done! Created '{title}'.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python generate_asset.py <art_id> <title> <category> <prompt_subject>")
    else:
        generate_and_process(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
