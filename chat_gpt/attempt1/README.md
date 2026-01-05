## Smart Asset Resizer (Computer Vision Based)

This tool converts a single 1080x1080 master creative into multiple platform-specific assets using saliency detection, contour-based object identification, and importance-aware cropping. It preserves key visual elements such as text, logos, and CTAs without clipping or padding. The system is modular and designed to be extended with ML-based detectors in production.

### Run
```bash
pip install -r requirements.txt
python main.py
