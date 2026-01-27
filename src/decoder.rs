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

/// Conceptual glyph grid dimensions (5x5)
pub const GLYPH_WIDTH: u32 = 5;
pub const GLYPH_HEIGHT: u32 = 5;

/// A 5x5 bitmap represented as 25 bits (each element is 0 or 1)
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
        let empty_bitmap = vec![0u8; (GLYPH_WIDTH * GLYPH_HEIGHT) as usize];
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

/// Extract a character region as a 5x5 bitmap by averaging pixel luminosity
pub fn extract_character(img: &image::GrayImage, start_x: u32, start_y: u32) -> Bitmap {
    let mut bitmap = Vec::with_capacity((GLYPH_WIDTH * GLYPH_HEIGHT) as usize);

    // The glyph is conceptually divided into a 5x5 grid of pixels that are black or white. We are
    // calling these conceptual "pixels" cells here, to distinguish them from the actual pixels of
    // the glyph representation. Since the glyph dimensions might not be divisible by 5, we use
    // floating point sizes for the cell dimensions.
    let cell_width = CHAR_WIDTH as f64 / GLYPH_WIDTH as f64;
    let cell_height = CHAR_HEIGHT as f64 / GLYPH_HEIGHT as f64;

    // Divide the 22x22 pixel region into a 5x5 grid of cells
    for cell_y in 0..GLYPH_HEIGHT {
        for cell_x in 0..GLYPH_WIDTH {
            // Calculate pixel boundaries for this cell. Cell sizes will not be integers when the
            // glyph width/height are not divisible by 5. We try to be conservative in
            // approximating the cell boundaries, so we use ceil and floor to leave out boundary
            // pixels that may or may not belong to the cell we're scanning. This should be fine,
            // we don't need the entire cell to classify its color.
            let px_start = (cell_x as f64 * cell_width).ceil() as u32;
            let px_end = ((cell_x + 1) as f64 * cell_width).floor() as u32;
            let py_start = (cell_y as f64 * cell_height).ceil() as u32;
            let py_end = ((cell_y + 1) as f64 * cell_height).floor() as u32;

            // Calculate mean luminosity for this cell
            let mut total_luma: u32 = 0;
            let mut pixel_count: u32 = 0;

            for py in py_start..py_end {
                for px in px_start..px_end {
                    let pixel_x = start_x + px;
                    let pixel_y = start_y + py;

                    if pixel_x < img.width() && pixel_y < img.height() {
                        let Luma([luma]) = *img.get_pixel(pixel_x, pixel_y);
                        total_luma += luma as u32;
                        pixel_count += 1;
                    }
                }
            }

            // Classify cell by mean luminosity (>= 128 -> white, < 128 -> black)
            let value = if pixel_count > 0 {
                let avg_luma = total_luma / pixel_count;
                if avg_luma >= 128 { 1 } else { 0 }
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

const MARGIN: usize = 4;
const PIXELS_PER_CELL: usize = 5;
const INNER_SIZE: usize = GLYPH_WIDTH as usize * PIXELS_PER_CELL; // 25
/// Size of glyph texture with margin
pub const GLYPH_TEXTURE_SIZE: usize = INNER_SIZE + 2 * MARGIN; // 33

/// Convert a 5x5 bitmap to an RGBA image buffer for egui texture with black margin
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
                // Map display coordinates to 5x5 bitmap cell
                let bx = x - MARGIN;
                let by = y - MARGIN;
                // Scale from 22x22 display area to 5x5 bitmap
                let cell_x = (bx * GLYPH_WIDTH as usize) / INNER_SIZE;
                let cell_y = (by * GLYPH_HEIGHT as usize) / INNER_SIZE;
                let pixel = bitmap[cell_y * GLYPH_WIDTH as usize + cell_x];
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
