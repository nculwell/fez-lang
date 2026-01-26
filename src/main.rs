use image::Luma;
use std::collections::HashMap;
use std::env;

/// A 22x22 bitmap represented as 484 bits (each element is 0 or 1)
type Bitmap = Vec<u8>;

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
fn parse_image(path: &str) -> Result<Vec<Vec<u32>>, image::ImageError> {
    let img = image::open(path)?;
    let gray = img.to_luma8();

    let width = gray.width();
    let height = gray.height();

    // Character dimensions and gaps
    const CHAR_WIDTH: u32 = 22;
    const CHAR_HEIGHT: u32 = 22;
    const H_GAP: u32 = 4;
    const V_GAP: u32 = 7;
    const H_STRIDE: u32 = CHAR_WIDTH + H_GAP;  // 26
    const V_STRIDE: u32 = CHAR_HEIGHT + V_GAP; // 29

    // Calculate number of characters per row and number of rows
    let chars_per_row = (width + H_GAP) / H_STRIDE;
    let num_rows = (height + V_GAP) / V_STRIDE;

    let mut registry = CharacterRegistry::new();
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

    // Print character registry summary
    eprintln!("Found {} unique characters (including empty space as ID 0)", registry.next_id);

    Ok(lines)
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <image.jpg>", args[0]);
        std::process::exit(1);
    }

    let path = &args[1];

    match parse_image(path) {
        Ok(lines) => {
            for line in lines {
                let line_str: Vec<String> = line.iter().map(|id| id.to_string()).collect();
                println!("{}", line_str.join(" "));
            }
        }
        Err(e) => {
            eprintln!("Error processing image: {}", e);
            std::process::exit(1);
        }
    }
}
