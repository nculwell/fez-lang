mod decoder;

use decoder::{parse_image, bitmap_to_rgba, CharacterRegistry, GLYPH_TEXTURE_SIZE};
use eframe::egui;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const MAPPINGS_FILE: &str = "mappings.json";

struct GlyphMapperApp {
    registry: CharacterRegistry,
    mappings: HashMap<u32, char>,
    decoded_texts: Vec<(String, Vec<Vec<u32>>)>,
    selected_glyph: Option<u32>,
    glyph_textures: Vec<egui::TextureHandle>,
    textures_loaded: bool,
}

impl GlyphMapperApp {
    fn new() -> Self {
        let dir = Path::new("text_images");

        let mut registry = CharacterRegistry::new();
        let mut decoded_texts = Vec::new();

        if dir.exists() {
            let mut image_files: Vec<_> = fs::read_dir(dir)
                .map(|rd| {
                    rd.filter_map(|entry| entry.ok())
                        .map(|entry| entry.path())
                        .filter(|path| {
                            path.extension()
                                .map(|ext| {
                                    let ext = ext.to_ascii_lowercase();
                                    ext == "jpg" || ext == "jpeg" || ext == "png"
                                })
                                .unwrap_or(false)
                        })
                        .collect()
                })
                .unwrap_or_default();

            image_files.sort();

            for path in &image_files {
                let filename = path.file_name().unwrap().to_string_lossy().to_string();
                if let Ok(lines) = parse_image(path, &mut registry) {
                    decoded_texts.push((filename, lines));
                }
            }
        }

        let mappings = load_mappings();

        GlyphMapperApp {
            registry,
            mappings,
            decoded_texts,
            selected_glyph: None,
            glyph_textures: Vec::new(),
            textures_loaded: false,
        }
    }

    fn load_textures(&mut self, ctx: &egui::Context) {
        if self.textures_loaded {
            return;
        }

        for (id, bitmap) in self.registry.bitmaps.iter().enumerate() {
            let rgba = bitmap_to_rgba(bitmap);
            let size = GLYPH_TEXTURE_SIZE;
            let image = egui::ColorImage::from_rgba_unmultiplied([size, size], &rgba);
            let texture = ctx.load_texture(
                format!("glyph_{}", id),
                image,
                egui::TextureOptions::NEAREST,
            );
            self.glyph_textures.push(texture);
        }

        self.textures_loaded = true;
    }

    fn decode_id(&self, id: u32) -> String {
        if id == 0 {
            " ".to_string()
        } else if let Some(&c) = self.mappings.get(&id) {
            c.to_string()
        } else {
            "*".to_string()
        }
    }

    fn decode_line(&self, line: &[u32]) -> String {
        line.iter().map(|&id| self.decode_id(id)).collect()
    }
}

impl eframe::App for GlyphMapperApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.load_textures(ctx);

        // Handle keyboard input for mapping
        if let Some(selected_id) = self.selected_glyph {
            ctx.input(|i| {
                for event in &i.events {
                    if let egui::Event::Text(text) = event {
                        if let Some(c) = text.chars().next() {
                            self.mappings.insert(selected_id, c);
                        }
                    }
                }
            });
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Fez Glyph Mapper");
                ui.separator();
                if ui.button("Save Mappings").clicked() {
                    save_mappings(&self.mappings);
                }
                if ui.button("Load Mappings").clicked() {
                    self.mappings = load_mappings();
                }
                ui.separator();
                ui.label(format!("{} glyphs detected", self.registry.next_id));
            });
        });

        egui::SidePanel::left("glyph_panel")
            .default_width(350.0)
            .show(ctx, |ui| {
                ui.heading("Glyph Palette");
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    let cols = 8;
                    egui::Grid::new("glyph_grid")
                        .spacing([4.0, 4.0])
                        .show(ui, |ui| {
                            for (id, texture) in self.glyph_textures.iter().enumerate() {
                                let id = id as u32;
                                if id == 0 {
                                    // Skip the space glyph in the palette
                                    continue;
                                }
                                let is_selected = self.selected_glyph == Some(id);
                                let mapping_str = if let Some(&c) = self.mappings.get(&id) {
                                    c.to_string()
                                } else {
                                    "?".to_string()
                                };

                                ui.vertical(|ui| {
                                    let response = ui.add(
                                        egui::ImageButton::new(egui::Image::new(texture).fit_to_exact_size(egui::vec2(30.0, 30.0)))
                                            .selected(is_selected)
                                    );

                                    if response.clicked() {
                                        self.selected_glyph = Some(id);
                                    }

                                    ui.label(&mapping_str);
                                });

                                if id % cols == 0 {
                                    ui.end_row();
                                }
                            }
                        });

                    ui.add_space(20.0);
                    ui.separator();

                    if let Some(selected_id) = self.selected_glyph {
                        ui.horizontal(|ui| {
                            if let Some(texture) = self.glyph_textures.get(selected_id as usize) {
                                ui.add(egui::Image::new(texture).fit_to_exact_size(egui::vec2(44.0, 44.0)));
                            }
                            ui.vertical(|ui| {
                                let current_mapping = self.mappings.get(&selected_id)
                                    .map(|c| c.to_string())
                                    .unwrap_or_else(|| "(unmapped)".to_string());
                                ui.label(format!("Mapping: {}", current_mapping));
                                ui.label("Type a character to map");
                            });
                        });
                    } else {
                        ui.label("Click a glyph to select it");
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Decoded Text");
            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                for (filename, lines) in &self.decoded_texts {
                    ui.heading(format!("=== {} ===", filename));
                    ui.add_space(4.0);

                    ui.scope(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                        for line in lines {
                            ui.horizontal(|ui| {
                                // Left side: show glyphs with no spacing
                                for &id in line {
                                    if let Some(texture) = self.glyph_textures.get(id as usize) {
                                        ui.add(egui::Image::new(texture).fit_to_exact_size(egui::vec2(20.0, 20.0)));
                                    }
                                }

                                ui.add_space(20.0);
                                ui.separator();
                                ui.add_space(10.0);

                                // Right side: show decoded text
                                let decoded = self.decode_line(line);
                                ui.monospace(&decoded);
                            });
                        }
                    });

                    ui.add_space(12.0);
                }
            });
        });
    }
}

fn save_mappings(mappings: &HashMap<u32, char>) {
    let string_map: HashMap<String, String> = mappings
        .iter()
        .map(|(&id, &c)| (id.to_string(), c.to_string()))
        .collect();

    if let Ok(json) = serde_json::to_string_pretty(&string_map) {
        if let Err(e) = fs::write(MAPPINGS_FILE, json) {
            eprintln!("Failed to save mappings: {}", e);
        } else {
            eprintln!("Saved mappings to {}", MAPPINGS_FILE);
        }
    }
}

fn load_mappings() -> HashMap<u32, char> {
    let path = Path::new(MAPPINGS_FILE);
    if !path.exists() {
        return HashMap::new();
    }

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return HashMap::new(),
    };

    let string_map: HashMap<String, String> = match serde_json::from_str(&content) {
        Ok(m) => m,
        Err(_) => return HashMap::new(),
    };

    string_map
        .iter()
        .filter_map(|(id_str, char_str)| {
            let id: u32 = id_str.parse().ok()?;
            let c: char = char_str.chars().next()?;
            Some((id, c))
        })
        .collect()
}

fn setup_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    // Load Hack font
    let font_data = std::fs::read("fonts/Hack/Hack-Regular.ttf")
        .expect("Failed to read Hack-Regular.ttf");

    fonts.font_data.insert(
        "Hack".to_owned(),
        egui::FontData::from_owned(font_data).into(),
    );

    // Set Hack as the primary font for all text styles
    fonts.families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "Hack".to_owned());

    fonts.families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "Hack".to_owned());

    ctx.set_fonts(fonts);
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_title("Fez Glyph Mapper"),
        ..Default::default()
    };

    eframe::run_native(
        "Fez Glyph Mapper",
        options,
        Box::new(|cc| {
            setup_fonts(&cc.egui_ctx);
            Ok(Box::new(GlyphMapperApp::new()))
        }),
    )
}
