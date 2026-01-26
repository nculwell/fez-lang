use image::{GrayImage, Luma};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// A 22x22 bitmap represented as 484 bits (each element is 0 or 1)
type Bitmap = Vec<u8>;

/// 3x5 pixel font for digits 0-9
const DIGIT_FONT: [[u8; 15]; 10] = [
    [1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1], // 0
    [0,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,1], // 1
    [1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1], // 2
    [1,1,1, 0,0,1, 1,1,1, 0,0,1, 1,1,1], // 3
    [1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1], // 4
    [1,1,1, 1,0,0, 1,1,1, 0,0,1, 1,1,1], // 5
    [1,1,1, 1,0,0, 1,1,1, 1,0,1, 1,1,1], // 6
    [1,1,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1], // 7
    [1,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1], // 8
    [1,1,1, 1,0,1, 1,1,1, 0,0,1, 1,1,1], // 9
];

/// Character registry mapping bitmaps to IDs
struct CharacterRegistry {
    bitmaps: Vec<Bitmap>,
    lookup: HashMap<Bitmap, u32>,
    next_id: u32,
}

impl CharacterRegistry {
    fn new() -> Self {
        // ID 0 is reserved for empty (all zeros)
        let empty_bitmap = vec![0u8; 22 * 22];
        let mut lookup = HashMap::new();
        lookup.insert(empty_bitmap.clone(), 0);

        CharacterRegistry {
            bitmaps: vec![empty_bitmap],
            lookup,
            next_id: 1,
        }
    }

    /// Get existing ID or assign a new one for the given bitmap
    fn get_or_assign_id(&mut self, bitmap: Bitmap) -> u32 {
        if let Some(&id) = self.lookup.get(&bitmap) {
            return id;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.lookup.insert(bitmap.clone(), id);
        self.bitmaps.push(bitmap);
        id
    }
}

/// Extract a 22x22 character region as a bitmap
fn extract_character(img: &image::GrayImage, start_x: u32, start_y: u32) -> Bitmap {
    let mut bitmap = Vec::with_capacity(22 * 22);

    for y in 0..22 {
        for x in 0..22 {
            let pixel_x = start_x + x;
            let pixel_y = start_y + y;

            // Handle out-of-bounds by treating as black (0)
            let value = if pixel_x < img.width() && pixel_y < img.height() {
                let Luma([luma]) = *img.get_pixel(pixel_x, pixel_y);
                // Light pixels (>= 128) → 1, Dark pixels (< 128) → 0
                if luma >= 128 { 1 } else { 0 }
            } else {
                0
            };
            bitmap.push(value);
        }
    }

    bitmap
}

/// Parse an image and return character IDs organized by line
fn parse_image(path: &Path, registry: &mut CharacterRegistry) -> Result<Vec<Vec<u32>>, image::ImageError> {
    let img = image::open(path)?;
    let gray = img.to_luma8();

    let width = gray.width();
    let height = gray.height();

    // Character dimensions and gaps
    const CHAR_WIDTH: u32 = 22;
    const CHAR_HEIGHT: u32 = 22;
    const H_GAP: u32 = 4;
    const V_GAP: u32 = 6;
    const H_STRIDE: u32 = CHAR_WIDTH + H_GAP;  // 26
    const V_STRIDE: u32 = CHAR_HEIGHT + V_GAP; // 29

    // Calculate number of characters per row and number of rows
    let chars_per_row = (width + H_GAP) / H_STRIDE;
    let num_rows = (height + V_GAP) / V_STRIDE;

    let mut lines: Vec<Vec<u32>> = Vec::new();

    for row in 0..num_rows {
        let mut line_ids: Vec<u32> = Vec::new();
        let y = row * V_STRIDE;

        for col in 0..chars_per_row {
            let x = col * H_STRIDE;
            let bitmap = extract_character(&gray, x, y);
            let id = registry.get_or_assign_id(bitmap);
            line_ids.push(id);
        }

        lines.push(line_ids);
    }

    Ok(lines)
}

fn main() {
    let dir = Path::new("text_images");

    if !dir.exists() {
        eprintln!("Error: text_images directory not found");
        std::process::exit(1);
    }

    // Collect and sort image files
    let mut image_files: Vec<_> = fs::read_dir(dir)
        .expect("Failed to read text_images directory")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .map(|ext| ext.to_ascii_lowercase() == "jpg" || ext.to_ascii_lowercase() == "jpeg")
                .unwrap_or(false)
        })
        .collect();

    image_files.sort();

    if image_files.is_empty() {
        eprintln!("No JPG files found in text_images directory");
        std::process::exit(1);
    }

    // Shared registry across all images
    let mut registry = CharacterRegistry::new();

    for path in &image_files {
        let filename = path.file_name().unwrap().to_string_lossy();
        println!("=== {} ===", filename);

        match parse_image(path, &mut registry) {
            Ok(lines) => {
                for line in lines {
                    let line_str: Vec<String> = line.iter().map(|id| format!("{:2}", id)).collect();
                    println!("{}", line_str.join(" "));
                }
            }
            Err(e) => {
                eprintln!("Error processing {}: {}", filename, e);
            }
        }
        println!();
    }

    eprintln!("Found {} unique glyphs (including empty space as ID 0)", registry.next_id);

    // Generate debug PNG
    generate_glyph_map(&registry);
}

/// Draw a digit at the specified position in the image
fn draw_digit(img: &mut GrayImage, digit: u8, x: u32, y: u32) {
    let font = &DIGIT_FONT[digit as usize];
    for dy in 0..5 {
        for dx in 0..3 {
            if font[dy * 3 + dx] == 1 {
                img.put_pixel(x + dx as u32, y + dy as u32, Luma([255]));
            }
        }
    }
}

/// Draw a number (multiple digits) at the specified position
fn draw_number(img: &mut GrayImage, mut num: u32, x: u32, y: u32) {
    if num == 0 {
        draw_digit(img, 0, x, y);
        return;
    }

    // Count digits
    let mut digits = Vec::new();
    while num > 0 {
        digits.push((num % 10) as u8);
        num /= 10;
    }
    digits.reverse();

    // Draw each digit with 4-pixel spacing
    for (i, &d) in digits.iter().enumerate() {
        draw_digit(img, d, x + (i as u32) * 4, y);
    }
}

/// Generate a PNG showing all glyphs with their IDs
fn generate_glyph_map(registry: &CharacterRegistry) {
    const CELL_WIDTH: u32 = 28;   // 22 glyph + 6 padding
    const CELL_HEIGHT: u32 = 32;  // 22 glyph + 8 for ID + 2 padding
    const COLS: u32 = 16;

    let num_glyphs = registry.bitmaps.len() as u32;
    let rows = (num_glyphs + COLS - 1) / COLS;

    let img_width = COLS * CELL_WIDTH;
    let img_height = rows * CELL_HEIGHT;

    let mut img = GrayImage::new(img_width, img_height);

    // Fill with dark gray background
    for pixel in img.pixels_mut() {
        *pixel = Luma([32]);
    }

    for (id, bitmap) in registry.bitmaps.iter().enumerate() {
        let col = (id as u32) % COLS;
        let row = (id as u32) / COLS;

        let cell_x = col * CELL_WIDTH;
        let cell_y = row * CELL_HEIGHT;

        // Draw ID number at top of cell
        draw_number(&mut img, id as u32, cell_x + 2, cell_y + 1);

        // Draw glyph below the ID
        let glyph_x = cell_x + 3;
        let glyph_y = cell_y + 8;

        for gy in 0..22 {
            for gx in 0..22 {
                let pixel_val = bitmap[(gy * 22 + gx) as usize];
                if pixel_val == 1 {
                    img.put_pixel(glyph_x + gx as u32, glyph_y + gy as u32, Luma([255]));
                }
            }
        }
    }

    img.save("glyph_map.png").expect("Failed to save glyph_map.png");
    eprintln!("Generated glyph_map.png");
}
