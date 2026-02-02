use image::{DynamicImage, GrayImage, Luma};
use std::collections::HashMap;
use std::path::Path;

/// Result of grayscale conversion, indicating which method was used
enum GrayMethod {
    DominantChannel,
    Luminosity,
}

/// Convert an image to grayscale using a hybrid approach.
/// If one color channel dominates (more than 2x the others) for at least 25% of pixels,
/// use that channel. Otherwise, use standard luminosity.
fn to_gray_hybrid(img: &DynamicImage) -> (GrayImage, GrayMethod) {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let total_pixels = (width * height) as usize;

    // Count pixels where one channel is more than double the others
    let mut dominant_counts = [0usize; 3]; // R, G, B
    for pixel in rgb.pixels() {
        let r = pixel[0] as u16;
        let g = pixel[1] as u16;
        let b = pixel[2] as u16;

        // Check if R dominates (R > 2*G and R > 2*B)
        if r > 2 * g && r > 2 * b {
            dominant_counts[0] += 1;
        }
        // Check if G dominates
        else if g > 2 * r && g > 2 * b {
            dominant_counts[1] += 1;
        }
        // Check if B dominates
        else if b > 2 * r && b > 2 * g {
            dominant_counts[2] += 1;
        }
    }

    let threshold_count = total_pixels / 4; // 25%
    let dominant_channel = if dominant_counts[0] >= threshold_count {
        Some(0)
    } else if dominant_counts[1] >= threshold_count {
        Some(1)
    } else if dominant_counts[2] >= threshold_count {
        Some(2)
    } else {
        None
    };

    let mut gray = GrayImage::new(width, height);

    if let Some(channel) = dominant_channel {
        let channel_name = ["red", "green", "blue"][channel];
        let pct = dominant_counts[channel] * 100 / total_pixels;
        eprintln!(
            "  Color imbalance detected: {} dominant in {}% of pixels, using {} channel",
            channel_name, pct, channel_name
        );

        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                gray.put_pixel(x, y, Luma([pixel[channel]]));
            }
        }
        (gray, GrayMethod::DominantChannel)
    } else {
        eprintln!("  No color imbalance, using luminosity");

        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                // Standard luminosity: 0.299*R + 0.587*G + 0.114*B
                let luma = (0.299 * pixel[0] as f32
                    + 0.587 * pixel[1] as f32
                    + 0.114 * pixel[2] as f32) as u8;
                gray.put_pixel(x, y, Luma([luma]));
            }
        }
        (gray, GrayMethod::Luminosity)
    }
}

/// Detected layout parameters for an image
#[derive(Debug, Clone)]
struct ImageLayout {
    char_width: u32,
    char_height: u32,
    h_stride: u32,
    v_stride: u32,
    x_offset: u32,
    y_offset: u32,
    /// Detected column regions: (start_x, width) for each non-uniform column
    col_regions: Vec<(u32, u32)>,
    /// Detected row regions: (start_y, height) for each non-uniform row
    row_regions: Vec<(u32, u32)>,
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

/// Compute the optimal threshold using Otsu's method.
/// Finds the threshold that maximizes between-class variance.
fn otsu_threshold(img: &image::GrayImage) -> u8 {
    let total_pixels = (img.width() * img.height()) as f64;
    if total_pixels == 0.0 {
        return 128;
    }

    // Build histogram
    let mut histogram = [0u32; 256];
    for pixel in img.pixels() {
        let Luma([luma]) = *pixel;
        histogram[luma as usize] += 1;
    }

    // Precompute total sum of all pixel values
    let total_sum: f64 = histogram
        .iter()
        .enumerate()
        .map(|(i, &count)| i as f64 * count as f64)
        .sum();

    let mut best_threshold = 0u8;
    let mut best_variance = 0.0f64;
    let mut sum_bg = 0.0f64;
    let mut weight_bg = 0.0f64;

    for (t, &count) in histogram.iter().enumerate() {
        weight_bg += count as f64;
        if weight_bg == 0.0 {
            continue;
        }

        let weight_fg = total_pixels - weight_bg;
        if weight_fg == 0.0 {
            break;
        }

        sum_bg += t as f64 * count as f64;
        let mean_bg = sum_bg / weight_bg;
        let mean_fg = (total_sum - sum_bg) / weight_fg;

        let variance = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if variance > best_variance {
            best_variance = variance;
            best_threshold = t as u8;
        }
    }

    best_threshold
}

/// Find threshold as midpoint between the two histogram peaks.
fn histogram_peaks_threshold(img: &image::GrayImage) -> u8 {
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

    ((peak1_idx as u16 + peak2_idx as u16) / 2) as u8
}

/// Detect the layout of glyphs in an image by finding uniform rows and columns
fn detect_layout(img: &image::GrayImage, method: &GrayMethod) -> Option<ImageLayout> {
    let width = img.width();
    let height = img.height();

    // Choose threshold method based on how the image was converted to grayscale
    let luma_threshold = match method {
        GrayMethod::DominantChannel => {
            let t = otsu_threshold(img);
            eprintln!("  Threshold (Otsu): {}", t);
            t
        }
        GrayMethod::Luminosity => {
            let t = histogram_peaks_threshold(img);
            eprintln!("  Threshold (histogram peaks): {}", t);
            t
        }
    };

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

    // Calculate stride from spacing between glyphs (if there are multiple)
    let h_stride = if col_regions.len() > 1 {
        col_regions[1].0 - col_regions[0].0
    } else {
        char_width
    };
    let v_stride = if row_regions.len() > 1 {
        row_regions[1].0 - row_regions[0].0
    } else {
        char_height
    };

    Some(ImageLayout {
        char_width,
        char_height,
        h_stride,
        v_stride,
        x_offset,
        y_offset,
        col_regions,
        row_regions,
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
    let (gray, method) = to_gray_hybrid(&img);
    eprintln!("  Image size: {}x{}", gray.width(), gray.height());

    // Detect layout by finding uniform rows/columns that separate glyphs
    let layout = match detect_layout(&gray, &method) {
        Some(l) => l,
        None => {
            eprintln!("  No glyphs detected in image");
            return Ok(Vec::new());
        }
    };

    eprintln!(
        "  Layout: {}x{} detected regions, char size {}x{}, stride {}x{}, offset ({}, {})",
        layout.col_regions.len(),
        layout.row_regions.len(),
        layout.char_width,
        layout.char_height,
        layout.h_stride,
        layout.v_stride,
        layout.x_offset,
        layout.y_offset
    );

    let h_stride = layout.h_stride;
    let v_stride = layout.v_stride;

    // Calculate expected grid positions based on stride, detecting spaces in gaps
    // A position has a glyph if there's a detected region near it; otherwise it's a space
    let tolerance = layout.char_height / 2;

    // Find row positions: either detected regions or spaces where gaps exist
    let mut row_positions: Vec<Option<u32>> = Vec::new(); // None = space row, Some(y) = glyph row
    let mut region_idx = 0;
    let mut expected_y = layout.y_offset;
    while expected_y + layout.char_height <= gray.height() {
        // Check if there's a detected row region near this expected position
        let found_region = if region_idx < layout.row_regions.len() {
            let (region_y, _) = layout.row_regions[region_idx];
            if region_y <= expected_y + tolerance {
                region_idx += 1;
                Some(region_y)
            } else {
                None // Gap - this is a space row
            }
        } else {
            None // No more regions
        };
        row_positions.push(found_region);
        expected_y += v_stride;
    }

    // Find column positions similarly
    let mut col_positions: Vec<Option<u32>> = Vec::new(); // None = space col, Some(x) = glyph col
    let mut region_idx = 0;
    let mut expected_x = layout.x_offset;
    let col_tolerance = layout.char_width / 2;
    while expected_x + layout.char_width <= gray.width() {
        let found_region = if region_idx < layout.col_regions.len() {
            let (region_x, _) = layout.col_regions[region_idx];
            if region_x <= expected_x + col_tolerance {
                region_idx += 1;
                Some(region_x)
            } else {
                None
            }
        } else {
            None
        };
        col_positions.push(found_region);
        expected_x += h_stride;
    }

    eprintln!(
        "  Grid: {}x{} (with spaces)",
        col_positions.len(),
        row_positions.len()
    );

    // First pass: extract all bitmaps from the image
    let empty_bitmap = vec![0u8; (GLYPH_WIDTH * GLYPH_HEIGHT) as usize];
    let mut bitmaps: Vec<Vec<Bitmap>> = Vec::new();
    for row_pos in &row_positions {
        let mut line_bitmaps: Vec<Bitmap> = Vec::new();

        for col_pos in &col_positions {
            let bitmap = match (row_pos, col_pos) {
                (Some(y), Some(x)) => extract_character(
                    &gray,
                    *x,
                    *y,
                    layout.char_width,
                    layout.char_height,
                    layout.luma_threshold,
                ),
                _ => empty_bitmap.clone(), // Space
            };
            line_bitmaps.push(bitmap);
        }
        bitmaps.push(line_bitmaps);
    }

    let total_glyphs: usize = bitmaps.iter().map(|line| line.len()).sum();

    // // Print luma grid for debugging if very few glyphs were found
    // if total_glyphs < 5 {
    //     print_luma_grid(&gray);
    // }

    // If the registry already has characters, check if this image is inverted
    // Only consider non-empty bitmaps (spaces always match and shouldn't affect inversion detection)
    if registry.next_id > 1 {
        let non_empty_bitmaps: Vec<&Bitmap> = bitmaps
            .iter()
            .flatten()
            .filter(|b| **b != empty_bitmap)
            .collect();

        let matches = non_empty_bitmaps
            .iter()
            .filter(|b| registry.contains(b))
            .count();

        eprintln!(
            "  Matches with existing glyphs: {}/{}",
            matches, non_empty_bitmaps.len()
        );

        if matches == 0 && !non_empty_bitmaps.is_empty() {
            eprintln!("  Flipping all bitmaps (image appears inverted)");
            // No matches found - flip all non-empty bitmaps
            for line in &mut bitmaps {
                for bitmap in line {
                    if *bitmap != empty_bitmap {
                        *bitmap = flip_bitmap(bitmap);
                    }
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
