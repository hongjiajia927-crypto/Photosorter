# PhotoSorter — Code Literacy Project (Weeks 7–11)

PhotoSorter is a Python desktop prototype for organising photos by **date** or by **similarity of detected human faces**. It includes a Tkinter graphical interface so users can select input and output folders without editing Python paths.

> **Prototype limitation:** Face grouping is experimental. In one test, photos showing four people were split into nine groups, even after adjusting the distance threshold. The application does not guarantee identity accuracy.

## Features

- **Week 7 — Image basics:** Use Pillow to read image format, dimensions, colour mode, and verify files.
- **Week 8 — EXIF metadata:** Read photo dates; fall back to file modification dates if EXIF date is unavailable.
- **Week 9 — Sorting:** Copy photos into `YYYY-MM-DD` folders, or detect human faces with DeepFace/FaceNet512 and group their embeddings by cosine distance using average-link clustering.
- **Week 10 — GUI:** Tkinter folder selection, sorting modes, progress, and face distance threshold.
- **Week 11 — Testing and safeguards:** SHA-256 detection of byte-identical files, logging, error handling, and separate fresh destinations for face sorting.

**Original photos are copied, not moved or deleted.** A group photo may be copied to multiple person folders if it contains multiple detected faces.

## Requirements

- Python 3.10+ (compatibility depends on installed DeepFace/TensorFlow versions)
- macOS with Tkinter support (the project was developed for Mac; other desktop systems may also work)
- Pillow for both modes
- DeepFace and OpenCV for face mode

Install dependencies in a Python environment compatible with DeepFace:

```bash
python3 -m pip install -r requirements.txt
```

DeepFace may download model weights the first time face sorting is used. Installation can take some time, and some Python versions may not be supported by its dependencies.

## Run

```bash
python3 PhotoSorter.py
```

On the author's Mac, Python was invoked using `/usr/local/bin/python3`.

1. Click **Browse** to choose the source photo folder.
2. Choose a separate destination folder.
3. Select **Sort by Date** or **Sort by Face**.
4. For face mode, use a **new, empty destination folder** each time; adjust the face distance threshold only if needed.
5. Start sorting and inspect the result folders.

## Example outputs

Date sorting:

```text
sorted_photos/
├── 2026-10-04/
└── 2026-10-15/
```

Face sorting:

```text
face_sorted_photos/
├── Person_1/
├── Person_2/
└── No_Faces/
```

The number of `Person_N` folders is determined by the algorithm and may not match the actual number of people.

## Known limitations and future work

- Face detection and clustering can split one person into multiple groups or merge different people.
- Similar-looking photos are **not** treated as duplicates unless their file bytes match exactly.
- File modification dates are not necessarily the original photo-taking dates.
- A future version could include manual merging of person groups, more thorough accuracy evaluation, and improved face detection.

## Development and AI use

This project was developed iteratively across Weeks 7–11. AI assistance was used to help generate and revise code and explain debugging approaches. Features were tested by running the application and inspecting output folders. The face-grouping limitation is reported rather than presented as resolved.

## Files

- `PhotoSorter.py` — main application and GUI
- `requirements.txt` — Python dependencies
- `.gitignore` — files excluded from Git

## Privacy

Do not upload private photo collections, face embeddings, or log files to a public repository. Test images are intentionally not included.
