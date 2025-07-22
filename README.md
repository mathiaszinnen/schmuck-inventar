# schmuck-inventar

This is a tool to extract structured data from fixed layout documents such as inventory cards, inventory books or tables. It requires layout definitions provided in `schmuck_inventar/config/regions.yaml`. Currently, it is optimized to work with a collection of jewelry from the Schmuckmuseum Pforzheim, but with minor adaptations it should be adaptable to other use cases. 

Specifically, to adapt this for your own data:
- Define a different document layout in `schmuck_inventar/config/regions.yaml`
- Implement your own post-processing by subclassing the abstract PostProcessor class in `schmuck_inventar/postprocessor.py`

## Installation

Set up your environment using your favourite environment management tool. We recommend [venv](https://docs.python.org/3/library/venv.html). 

Install using `pip install schmuck-inventar`

## Usage

Invoke using `schmuck-inventar <input folder>` with input folder specifying a path to the inventory cards to be extracted. Results will be saved to `output/` relative to the current working directory.

### Command Line Arguments

```
schmuck-inventar <input_dir> [options]
```

**Required Arguments:**
- `input_dir`: Path to the input directory containing files to process

**Optional Arguments:**
- `--output_dir`: Path to the output directory (default: `./output`)
- `--layout_config`: Path to the layout configuration file (default: `config/regions.yaml`)
- `--ocr_engine`: OCR engine to use (choices: `auto`, `ocrmac`, `pero`, `mistral`, `dummy`; default: `auto`)
- `--eval`: Run in evaluation mode (uses dummy detector and benchmarking postprocessor)

### Supported OCR Engines

The tool supports multiple OCR engines with automatic platform-based selection:

- **PERO OCR** (`pero`): Default for non-macOS platforms
- **Apple Vision API** (`ocrmac`): Default for macOS platforms  
- **Mistral OCR** (`mistral`): AI-powered OCR using Mistral models
- **Dummy** (`dummy`): For development and testing purposes
- **Auto** (`auto`): Automatically selects `ocrmac` on macOS, `pero` on other platforms

### Installing Optional Dependencies

Different OCR engines require additional dependencies:

**For PERO OCR (non-macOS default):**
```bash
pip install schmuck-inventar[pero]
```

**For Mistral OCR:**
```bash
pip install schmuck-inventar[mistral]
```

You'll also need to obtain a Mistral API key and store it in a `.env` file in your project root:

1. Copy `.env.template` to `.env`:
   ```bash
   cp .env.template .env
   ```

2. Edit `.env` and replace `INSERT_YOUR_KEY_HERE` with your actual Mistral API key:
   ```
   MISTRAL_API_KEY=your_actual_api_key_here
   ```

**Note:** Never share your API key publicly or commit it to version control.

**For Apple Vision API (macOS only):**
The `ocrmac` package is automatically installed on macOS systems. No additional installation required.

**Note:** The `auto` engine selection will use Apple Vision API on macOS (if available) and PERO OCR on other platforms.

## Configuration

### Region Definition

Regions (in relative coordinates) and field names are stored in `schmuck_inventar/config/regions.yaml` and can be adapted according to the use case.

The region definition format uses normalized coordinates where each region is defined as `[x1, y1, x2, y2]` with values between 0 and 1:
- `x1, y1`: Top-left corner coordinates (relative to document width and height)
- `x2, y2`: Bottom-right corner coordinates (relative to document width and height)

#### Example Region Configuration

```yaml
regions:
  Gegenstand: [0, 0, 1, 0.1]           # Full width header at top
  Inv. Nr.: [0.044, 0.08, 0.275, 0.145] # Inventory number field
  Herkunft: [0.2737, 0.0806, 0.5054, 0.237] # Origin field
  Foto Notes: [0.5054, 0.0806, 1, 0.237] # Photo notes field
  Standort: [0.047, 0.135, 0.275, 0.235] # Location field
  Material: [0.047, 0.237, 0.275, 0.435] # Material field
  Datierung: [0.275, 0.237, 0.505, 0.317] # Dating field
  Maße: [0.275, 0.317, 0.505, 0.435]   # Measurements field
  erworben von: [0.047, 0.435, 0.505, 0.521] # Acquired from field
  Beschreibung: [0.047, 0.521, 0.505, 0.794] # Description field
  Ausstellungen: [0.047, 0.795, 0.505, 1] # Exhibitions field
  am: [0.505, 0.435, 0.707, 0.53]      # Date acquired field
  Preis: [0.707, 0.435, 0.869, 0.53]   # Price field
  Vers.-Wert: [0.870, 0.435, 1.0, 0.53] # Insurance value field
  Literatur: [0.50, 0.53, 1.0, 1.0]    # Literature field

custom_header_mappings:
  am: 'erworben am'  # Maps "am" field to "erworben am" in output
