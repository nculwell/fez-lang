use image::Luma;
use std::collections::HashMap;
use std::path::Path;

/// Character dimensions and gaps
pub const CHAR_WIDTH: u32 = 22;
pub const CHAR_HEIGHT: u32 = 22;
pub const H_GAP: u32 = 4;
pub const V_GAP: u32 = 6;
pub const H_STRIDE: u32 = CHAR_WIDTH + H_GAP;  // 26
pub const V_STRIDE: u32 = CHAR_HEIGHT + V_GAP; // 29

/// A 22x22 bitmap represented as 484 bits (each element is 0 or 1)
pub type Bitmap = Vec<u8>;

/// Character registry mapping bitmaps to IDs
pub struct CharacterRegistry {
    pub bitmaps: Vec<Bitmap>,
    lookup: HashMap<Bitmap, u32>,
    pub next_id: u32,
}

impl CharacterRegistry {
    pub fn new() -> Self {
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
    pub fn get_or_assign_id(&mut self, bitmap: Bitmap) -> u32 {
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
pub fn extract_character(img: &image::GrayImage, start_x: u32, start_y: u32) -> Bitmap {
    let mut bitmap = Vec::with_capacity(22 * 22);

    for y in 0..22 {
        for x in 0..22 {
            let pixel_x = start_x + x;
            let pixel_y = start_y + y;

            // Handle out-of-bounds by treating as black (0)
            let value = if pixel_x < img.width() && pixel_y < img.height() {
                let Luma([luma]) = *img.get_pixel(pixel_x, pixel_y);
                // Light pixels (>= 128) -> 1, Dark pixels (< 128) -> 0
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
pub fn parse_image(path: &Path, registry: &mut CharacterRegistry) -> Result<Vec<Vec<u32>>, image::ImageError> {
    let img = image::open(path)?;
    let gray = img.to_luma8();

    let width = gray.width();
    let height = gray.height();

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

/// Size of glyph texture with margin (22 + 4*2 = 30)
pub const GLYPH_TEXTURE_SIZE: usize = 30;
const MARGIN: usize = 4;

/// Convert a bitmap to an RGBA image buffer for egui texture with black margin
pub fn bitmap_to_rgba(bitmap: &Bitmap) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(GLYPH_TEXTURE_SIZE * GLYPH_TEXTURE_SIZE * 4);

    for y in 0..GLYPH_TEXTURE_SIZE {
        for x in 0..GLYPH_TEXTURE_SIZE {
            // Check if we're in the margin area
            let in_margin = x < MARGIN || x >= GLYPH_TEXTURE_SIZE - MARGIN
                         || y < MARGIN || y >= GLYPH_TEXTURE_SIZE - MARGIN;

            let val = if in_margin {
                0 // Black margin
            } else {
                // Map to original bitmap coordinates
                let bx = x - MARGIN;
                let by = y - MARGIN;
                let pixel = bitmap[by * 22 + bx];
                if pixel == 1 { 255 } else { 0 }
            };

            rgba.push(val); // R
            rgba.push(val); // G
            rgba.push(val); // B
            rgba.push(255); // A
        }
    }
    rgba
}
