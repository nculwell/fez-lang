use image::Luma;
use std::collections::HashMap;
use std::path::Path;

/// Detected layout parameters for an image
#[derive(Debug, Clone)]
struct ImageLayout {
    char_width: u32,
    char_height: u32,
    h_gap: u32,
    v_gap: u32,
    x_offset: u32,
    y_offset: u32,
    chars_per_row: u32,
    num_rows: u32,
    luma_threshold: u8,
}

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

    /// Check if a bitmap exists in the registry
    pub fn contains(&self, bitmap: &Bitmap) -> bool {
        self.lookup.contains_key(bitmap)
    }
}

/// Flip all bits in a bitmap (0 -> 1, 1 -> 0)
fn flip_bitmap(bitmap: &Bitmap) -> Bitmap {
    bitmap.iter().map(|&b| if b == 0 { 1 } else { 0 }).collect()
}

/// Check if a row is uniform (i.e. if all pixels have the same color)
fn is_uniform_row(img: &image::GrayImage, y: u32, luma_threshold: u8) -> bool {
    if img.width() == 0 {
        return true;
    }
    let Luma([first_pixel_luma]) = *img.get_pixel(0, y);
    let first_pixel_is_white = first_pixel_luma >= luma_threshold;
    for x in 1..img.width() {
        let Luma([current_pixel_luma]) = *img.get_pixel(x, y);
        let current_pixel_is_white = current_pixel_luma >= luma_threshold;
        if current_pixel_is_white != first_pixel_is_white {
            return false;
        }
    }
    // If we didn't find any pixels that were a different color, then we conclude that the row is
    // uniform.
    true
}

/// Check if a column is uniform (i.e. if all pixels have the same color)
fn is_uniform_col(img: &image::GrayImage, x: u32, luma_threshold: u8) -> bool {
    if img.height() == 0 {
        return true;
    }
    let Luma([first_pixel_luma]) = *img.get_pixel(x, 0);
    let first_pixel_is_white = first_pixel_luma >= luma_threshold;
    for y in 1..img.height() {
        let Luma([current_pixel_luma]) = *img.get_pixel(x, y);
        let current_pixel_is_white = current_pixel_luma >= luma_threshold;
        if current_pixel_is_white != first_pixel_is_white {
            return false;
        }
    }
    // If we didn't find any pixels that were a different color, then we conclude that the column
    // is uniform.
    true
}

/// Find runs of non-uniform rows/columns (glyph regions) separated by uniform ones (gaps)
/// Returns a list of (start, length) for each glyph region
fn find_glyph_regions(uniform: &[bool]) -> Vec<(u32, u32)> {
    let mut regions = Vec::new();

    // let mut prev_was_uniform = true;
    // let mut glyph_region_start: usize = 0;
    // for i in 0..uniform.len() {
    //     if uniform[i] {
    //         if prev_was_uniform {
    //             // do nothing
    //         } else {
    //             let glyph_region_end = i;
    //             regions.push((
    //                 glyph_region_start as u32,
    //                 (glyph_region_end - glyph_region_start) as u32,
    //             ));
    //         }
    //     } else {
    //         if !prev_was_uniform {
    //             // do nothing
    //         } else {
    //             glyph_region_start = i;
    //         }
    //     }
    //     prev_was_uniform = uniform[i];
    // }
    // if !prev_was_uniform {
    //     let glyph_region_end = uniform.len();
    //     regions.push((
    //         glyph_region_start as u32,
    //         (glyph_region_end - glyph_region_start) as u32,
    //     ));
    // }

    let mut i = 0;
    while i < uniform.len() {
        // Skip uniform (gap) region
        while i < uniform.len() && uniform[i] {
            i += 1;
        }
        if i >= uniform.len() {
            break;
        }
        // Found start of a glyph region
        let start = i;
        while i < uniform.len() && !uniform[i] {
            i += 1;
        }
        regions.push((start as u32, (i - start) as u32));
    }

    regions
}

/// Find threshold by locating the two histogram peaks (most common pixel values)
/// and returning their midpoint. This works well for synthetic images with two
/// distinct foreground/background colors.
fn histogram_peaks_threshold(img: &image::GrayImage) -> (u8, u8, u8) {
    // Build histogram
    let mut histogram = [0u32; 256];
    for pixel in img.pixels() {
        let Luma([luma]) = *pixel;
        histogram[luma as usize] += 1;
    }

    // Find the two highest peaks
    let mut peak1_idx = 0usize;
    let mut peak1_count = 0u32;
    let mut peak2_idx = 0usize;
    let mut peak2_count = 0u32;

    for (i, &count) in histogram.iter().enumerate() {
        if count > peak1_count {
            peak2_idx = peak1_idx;
            peak2_count = peak1_count;
            peak1_idx = i;
            peak1_count = count;
        } else if count > peak2_count {
            peak2_idx = i;
            peak2_count = count;
        }
    }

    // Ensure peak1 is the lower value
    let (low_peak, high_peak) = if peak1_idx < peak2_idx {
        (peak1_idx as u8, peak2_idx as u8)
    } else {
        (peak2_idx as u8, peak1_idx as u8)
    };

    let threshold = ((low_peak as u16 + high_peak as u16) / 2) as u8;
    (low_peak, high_peak, threshold)
}

/// Detect the layout of glyphs in an image by finding uniform rows and columns
fn detect_layout(img: &image::GrayImage) -> Option<ImageLayout> {
    let width = img.width();
    let height = img.height();

    // Find threshold using histogram peaks method
    let (low_peak, high_peak, luma_threshold) = histogram_peaks_threshold(img);
    eprintln!(
        "  Histogram peaks: {}, {}, threshold: {}",
        low_peak, high_peak, luma_threshold
    );

    // Find uniform rows and columns
    let uniform_rows: Vec<bool> = (0..height)
        .map(|y| is_uniform_row(img, y, luma_threshold))
        .collect();
    let uniform_cols: Vec<bool> = (0..width)
        .map(|x| is_uniform_col(img, x, luma_threshold))
        .collect();

    // Find glyph regions
    let row_regions = find_glyph_regions(&uniform_rows);
    let col_regions = find_glyph_regions(&uniform_cols);

    if row_regions.is_empty() || col_regions.is_empty() {
        return None;
    }

    // Extract layout parameters from the first glyph
    let x_offset = col_regions[0].0;
    let y_offset = row_regions[0].0;
    let char_width = col_regions[0].1;
    let char_height = row_regions[0].1;

    // Calculate gaps from spacing between glyphs (if there are multiple)
    let h_gap = if col_regions.len() > 1 {
        col_regions[1].0 - col_regions[0].0 - char_width
    } else {
        0
    };
    let v_gap = if row_regions.len() > 1 {
        row_regions[1].0 - row_regions[0].0 - char_height
    } else {
        0
    };

    Some(ImageLayout {
        char_width,
        char_height,
        h_gap,
        v_gap,
        x_offset,
        y_offset,
        chars_per_row: col_regions.len() as u32,
        num_rows: row_regions.len() as u32,
        luma_threshold,
    })
}

/// Print a grid of luminosity values for debugging (upper-left 64x64 block)
fn print_luma_grid(img: &image::GrayImage) {
    let max_size = 64;
    let width = img.width().min(max_size);
    let height = img.height().min(max_size);
    eprintln!("  Luma grid ({}x{}):", width, height);
    for y in 0..height {
        eprint!("    ");
        for x in 0..width {
            let Luma([luma]) = *img.get_pixel(x, y);
            eprint!("{:02x} ", luma);
        }
        eprintln!();
    }
}

/// Extract a character region as a 5x5 bitmap by averaging pixel luminosity
fn extract_character(
    img: &image::GrayImage,
    start_x: u32,
    start_y: u32,
    char_width: u32,
    char_height: u32,
    luma_threshold: u8,
) -> Bitmap {
    let mut bitmap = Vec::with_capacity((GLYPH_WIDTH * GLYPH_HEIGHT) as usize);

    // The glyph is conceptually divided into a 5x5 grid of pixels that are black or white. We are
    // calling these conceptual "pixels" cells here, to distinguish them from the actual pixels of
    // the glyph representation. Since the glyph dimensions might not be divisible by 5, we use
    // floating point sizes for the cell dimensions.
    let cell_width = char_width as f64 / GLYPH_WIDTH as f64;
    let cell_height = char_height as f64 / GLYPH_HEIGHT as f64;

    // Divide the pixel region into a 5x5 grid of cells
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

            // Classify cell by mean luminosity
            let value = if pixel_count > 0 {
                let mean_luma = (total_luma / pixel_count) as u8;
                if mean_luma >= luma_threshold {
                    1
                } else {
                    0
                }
            } else {
                0
            };
            bitmap.push(value);
        }
    }

    bitmap
}

/// Parse an image and return character IDs organized by line
pub fn parse_image(
    path: &Path,
    registry: &mut CharacterRegistry,
) -> Result<Vec<Vec<u32>>, image::ImageError> {
    eprintln!("Loading image: {}", path.display());
    let img = image::open(path)?;
    let gray = img.to_luma8();
    eprintln!("  Image size: {}x{}", gray.width(), gray.height());

    // Detect layout by finding uniform rows/columns that separate glyphs
    let layout = match detect_layout(&gray) {
        Some(l) => l,
        None => {
            eprintln!("  No glyphs detected in image");
            return Ok(Vec::new());
        }
    };

    eprintln!(
        "  Layout: {}x{} glyphs, char size {}x{}, gap {}x{}, offset ({}, {})",
        layout.chars_per_row,
        layout.num_rows,
        layout.char_width,
        layout.char_height,
        layout.h_gap,
        layout.v_gap,
        layout.x_offset,
        layout.y_offset
    );

    let h_stride = layout.char_width + layout.h_gap;
    let v_stride = layout.char_height + layout.v_gap;

    // First pass: extract all bitmaps from the image
    let mut bitmaps: Vec<Vec<Bitmap>> = Vec::new();
    for row in 0..layout.num_rows {
        let mut line_bitmaps: Vec<Bitmap> = Vec::new();
        let y = layout.y_offset + row * v_stride;

        for col in 0..layout.chars_per_row {
            let x = layout.x_offset + col * h_stride;
            let bitmap = extract_character(
                &gray,
                x,
                y,
                layout.char_width,
                layout.char_height,
                layout.luma_threshold,
            );
            line_bitmaps.push(bitmap);
        }
        bitmaps.push(line_bitmaps);
    }

    let total_glyphs: usize = bitmaps.iter().map(|line| line.len()).sum();

    // Print luma grid for debugging if very few glyphs were found
    if total_glyphs < 5 {
        print_luma_grid(&gray);
    }

    // If the registry already has characters, check if this image is inverted
    if registry.next_id > 1 {
        let matches = bitmaps
            .iter()
            .flatten()
            .filter(|b| registry.contains(b))
            .count();

        eprintln!(
            "  Matches with existing glyphs: {}/{}",
            matches, total_glyphs
        );

        if matches == 0 {
            eprintln!("  Flipping all bitmaps (image appears inverted)");
            // No matches found - flip all bitmaps
            for line in &mut bitmaps {
                for bitmap in line {
                    *bitmap = flip_bitmap(bitmap);
                }
            }
        }
    }

    // Second pass: assign IDs to all bitmaps
    let glyphs_before = registry.next_id;
    let mut lines: Vec<Vec<u32>> = Vec::new();
    for line_bitmaps in bitmaps {
        let line_ids: Vec<u32> = line_bitmaps
            .into_iter()
            .map(|bitmap| registry.get_or_assign_id(bitmap))
            .collect();
        lines.push(line_ids);
    }
    let new_glyphs = registry.next_id - glyphs_before;
    eprintln!(
        "  Extracted {} glyphs ({} new, {} total in registry)",
        total_glyphs, new_glyphs, registry.next_id
    );

    Ok(lines)
}

const MARGIN: usize = 4;

/// Calculate the texture size for a given glyph size (glyph_size + 2*MARGIN)
pub fn texture_size(glyph_size: usize) -> usize {
    glyph_size + 2 * MARGIN
}

/// The 5x5 glyph cells have widths in the pattern 4-4-6-4-4 for a size-22 glyph.
/// This maps a pixel position to a cell index (0-4) for a given glyph size.
fn pixel_to_cell(pos: usize, glyph_size: usize) -> usize {
    // Cumulative boundaries scaled from the base pattern (0, 4, 8, 14, 18, 22)
    // For glyph_size S: boundary[i] = base[i] * S / 22
    const BASE: [usize; 6] = [0, 4, 8, 14, 18, 22];
    for cell in 0..5 {
        let boundary = BASE[cell + 1] * glyph_size / 22;
        if pos < boundary {
            return cell;
        }
    }
    4 // Fallback to last cell
}

/// Convert a 5x5 bitmap to an RGBA image buffer for egui texture with black margin
pub fn bitmap_to_rgba(bitmap: &Bitmap, glyph_size: usize) -> Vec<u8> {
    let tex_size = texture_size(glyph_size);
    let mut rgba = Vec::with_capacity(tex_size * tex_size * 4);

    for y in 0..tex_size {
        for x in 0..tex_size {
            // Check if we're in the margin area
            let in_margin =
                x < MARGIN || x >= tex_size - MARGIN || y < MARGIN || y >= tex_size - MARGIN;

            let val = if in_margin {
                0 // Black margin
            } else {
                // Map display coordinates to 5x5 bitmap cell using 4-4-6-4-4 pattern
                let bx = x - MARGIN;
                let by = y - MARGIN;
                let cell_x = pixel_to_cell(bx, glyph_size);
                let cell_y = pixel_to_cell(by, glyph_size);
                let pixel = bitmap[cell_y * GLYPH_WIDTH as usize + cell_x];
                if pixel == 1 {
                    255
                } else {
                    0
                }
            };

            rgba.push(val); // R
            rgba.push(val); // G
            rgba.push(val); // B
            rgba.push(255); // A
        }
    }
    rgba
}
