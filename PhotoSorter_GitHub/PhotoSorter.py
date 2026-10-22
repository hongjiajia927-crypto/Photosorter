"""PhotoSorter | Weeks 7–11 | Improved version

Week 7  - Read and validate image information with Pillow.
Week 8  - Extract EXIF dates; fall back to file modification dates.
Week 9  - Sort by date OR group human faces with FaceNet512 embeddings.
          Face mode extracts ALL faces first, then clusters by similarity.
Week 10 - Tkinter GUI with folder selection, sorting modes and progress.
Week 11 - Avoid identical files, distinguish detection/model errors,
          prevent mixing old face groups with new ones, and log errors.

Install: /usr/local/bin/python3 -m pip install Pillow deepface opencv-python
Run:     /usr/local/bin/python3 PhotoSorter_Week7_to_Week11_Improved.py

Face groups are estimates, not confirmed identities. A group photo may appear
in multiple Person_N folders, but only once within any one folder.
Original photos are never moved or deleted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import shutil
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ExifTags

# ========================== SHARED SETTINGS ================================
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
FACE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
DEFAULT_FACE_DISTANCE = 0.40  # Cosine distance: larger merges more faces.
MODEL_NAME = 'Facenet512'
MANIFEST_NAME = '.photosorter_face_manifest.json'
LOG_FILE = Path.home() / 'Desktop' / 'photosorter.log'


def configure_logging():
    """Week 11: Put logs somewhere writable rather than macOS root (/)."""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=str(LOG_FILE), level=logging.INFO,
                            format='%(asctime)s %(levelname)s %(message)s')
    except OSError:
        logging.basicConfig(level=logging.INFO)


def validate_folders(source: Path, destination: Path) -> tuple[Path, Path]:
    """Week 11: Reject missing folders and nested source/output paths."""
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_dir():
        raise ValueError('Please select an existing source folder.')
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate, non-nested folders.')
    destination.mkdir(parents=True, exist_ok=True)
    return source, destination


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_new(source: Path, folder: Path, digest: str, cache: dict) -> bool:
    """Week 11: Prevent byte-identical copies, even on subsequent runs."""
    folder.mkdir(parents=True, exist_ok=True)
    if folder not in cache:
        cache[folder] = {file_sha256(p) for p in folder.iterdir() if p.is_file()}
    if digest in cache[folder]:
        return False
    target = folder / source.name
    count = 1
    while target.exists():
        target = folder / f'{source.stem}_{count}{source.suffix}'
        count += 1
    shutil.copy2(source, target)
    cache[folder].add(digest)
    return True


# ========================= WEEK 7: IMAGE BASICS ============================
def get_image_info(path: Path) -> dict:
    """Read image format, dimensions, colour mode and verify integrity."""
    with Image.open(path) as image:
        info = {'format': image.format, 'size': image.size, 'mode': image.mode}
        image.verify()
    return info


# ========================= WEEK 8: EXIF METADATA ===========================
def get_exif_data(image: Image.Image) -> dict:
    """Read EXIF including DateTimeOriginal from the EXIF sub-IFD."""
    raw = image.getexif()
    values = {ExifTags.TAGS.get(tag, tag): value for tag, value in raw.items()}
    try:
        for tag, value in raw.get_ifd(ExifTags.IFD.Exif).items():
            values[ExifTags.TAGS.get(tag, tag)] = value
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return values


def get_photo_date(path: Path) -> tuple[str, str]:
    """Return (YYYY-MM-DD, EXIF/file date)."""
    try:
        with Image.open(path) as image:
            metadata = get_exif_data(image)
        for key in ('DateTimeOriginal', 'DateTimeDigitized', 'DateTime'):
            value = metadata.get(key)
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='ignore')
            if value:
                try:
                    dt = datetime.strptime(str(value).strip()[:19], '%Y:%m:%d %H:%M:%S')
                    return dt.strftime('%Y-%m-%d'), 'EXIF'
                except ValueError:
                    continue
    except (OSError, ValueError) as exc:
        logging.info('EXIF unavailable for %s: %s', path, exc)
    return datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d'), 'file date'


# ========================= WEEK 9: DATE SORTING ============================
def sort_photos_by_date(source: Path, destination: Path, progress=None) -> dict:
    source, destination = validate_folders(source, destination)
    files = sorted(p for p in source.iterdir() if p.is_file())
    result = {'files': len(files), 'sorted': 0, 'duplicates': 0,
              'skipped': 0, 'errors': 0, 'exif': 0, 'fallback': 0}
    seen, cache = set(), {}
    for i, photo in enumerate(files, 1):
        try:
            if photo.suffix.lower() not in IMAGE_EXTENSIONS:
                result['skipped'] += 1
                continue
            get_image_info(photo)
            digest = file_sha256(photo)
            if digest in seen:
                result['duplicates'] += 1
                continue
            seen.add(digest)
            date, method = get_photo_date(photo)
            if copy_if_new(photo, destination / date, digest, cache):
                result['sorted'] += 1
            else:
                result['duplicates'] += 1
            result['exif' if method == 'EXIF' else 'fallback'] += 1
        except Exception:
            result['errors'] += 1
            logging.exception('Date sorting error: %s', photo)
        finally:
            if progress:
                progress(i, len(files), photo.name)
    return result


# ========================= WEEK 9: FACE RECOGNITION ========================
class FaceNotDetected(Exception):
    """A photo contains no detectable HUMAN face (not a model failure)."""


def extract_faces(path: Path) -> list[list[float]]:
    """Extract FaceNet512 human face embeddings; load DeepFace only when needed."""
    try:
        from deepface import DeepFace
    except ImportError as exc:
        raise RuntimeError('DeepFace is missing. Install: python3 -m pip install deepface opencv-python') from exc
    try:
        faces = DeepFace.represent(img_path=str(path), model_name=MODEL_NAME,
                                   detector_backend='opencv', enforce_detection=True)
    except ValueError as exc:
        msg = str(exc).lower()
        if any(part in msg for part in ('face could not be detected', 'face cannot be detected',
                                        'face is not detected', 'no face detected',
                                        'face could not be found')):
            raise FaceNotDetected() from exc
        raise
    embeddings = []
    for face in faces:
        values = face.get('embedding')
        if values is None:
            continue
        vector = [float(x) for x in values]
        if vector and all(math.isfinite(x) for x in vector) and math.sqrt(sum(x*x for x in vector)) > 0:
            embeddings.append(vector)
    if not embeddings:
        raise RuntimeError(f'Model returned no usable embeddings for {path.name}')
    return embeddings


def cosine_distance(first: list[float], second: list[float]) -> float:
    """Cosine distance: 0=identical directions, larger=less similar."""
    if len(first) != len(second) or not first:
        raise ValueError('Incompatible face embeddings.')
    dot = sum(a*b for a, b in zip(first, second))
    n1 = math.sqrt(sum(a*a for a in first))
    n2 = math.sqrt(sum(b*b for b in second))
    if n1 == 0 or n2 == 0:
        raise ValueError('Zero-length face embedding.')
    return max(0.0, min(2.0, 1 - dot/(n1*n2)))


def cluster_faces(embeddings: list[list[float]], threshold: float) -> list[int]:
    """Average-link agglomerative clustering, no extra scikit-learn dependency.

    Unlike sequential first-match grouping, this compares ALL detected faces.
    Cluster labels are deterministic for a fixed ordered input set.
    """
    n = len(embeddings)
    if n == 0:
        return []
    distances = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            distances[i][j] = distances[j][i] = cosine_distance(embeddings[i], embeddings[j])
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                a, b = clusters[i], clusters[j]
                distance = sum(distances[x][y] for x in a for y in b) / (len(a) * len(b))
                candidate = (distance, i, j)
                if best is None or candidate < best:
                    best = candidate
        if best is None or best[0] > threshold:
            break
        _, i, j = best
        clusters[i] = sorted(clusters[i] + clusters[j])
        del clusters[j]
    clusters.sort(key=lambda cluster: min(cluster))
    labels = [0] * n
    for label, cluster in enumerate(clusters, 1):
        for index in cluster:
            labels[index] = label
    return labels


def sort_photos_by_face(source: Path, destination: Path, progress=None,
                        threshold: float = DEFAULT_FACE_DISTANCE) -> dict:
    """Cluster all human faces, then copy each distinct image to its group(s).

    For safety, face mode requires a NEW EMPTY output folder. This prevents
    mixing earlier (possibly incorrect) Person_N groups with new clustering.
    To rerun, choose another empty folder. No existing output is deleted.
    """
    source, destination = validate_folders(source, destination)
    if not 0.05 <= threshold <= 0.90:
        raise ValueError('Face distance threshold must be between 0.05 and 0.90.')
    if any(destination.iterdir()):
        raise ValueError('Face mode needs a NEW EMPTY destination folder. Old Person groups are not reused. Choose a new folder to avoid mixing results.')
    files = sorted(p for p in source.iterdir() if p.is_file())
    result = {'files': len(files), 'sorted': 0, 'duplicates': 0, 'skipped': 0,
              'errors': 0, 'people': 0, 'no_faces': 0, 'faces_detected': 0}
    seen_hashes, unique_images = set(), []
    embeddings, embedding_photo_ids, no_face_ids = [], [], []
    for i, photo in enumerate(files, 1):
        try:
            if photo.suffix.lower() not in FACE_EXTENSIONS:
                result['skipped'] += 1
                continue
            get_image_info(photo)
            digest = file_sha256(photo)
            if digest in seen_hashes:
                result['duplicates'] += 1
                continue
            seen_hashes.add(digest)
            image_id = len(unique_images)
            unique_images.append((photo, digest))
            try:
                photo_faces = extract_faces(photo)
            except FaceNotDetected:
                no_face_ids.append(image_id)
                continue
            for face in photo_faces:
                embeddings.append(face)
                embedding_photo_ids.append(image_id)
        except Exception as exc:
            result['errors'] += 1
            result.setdefault('first_error', f'{photo.name}: {exc}')
            logging.exception('Face extraction error: %s', photo)
        finally:
            if progress:
                progress(i, len(files), photo.name)

    # Clustering happens only after the entire input set has been analysed.
    labels = cluster_faces(embeddings, threshold)
    result['faces_detected'] = len(embeddings)
    result['people'] = len(set(labels))
    photo_groups: dict[int, set[int]] = {}
    for label, image_id in zip(labels, embedding_photo_ids):
        photo_groups.setdefault(label, set()).add(image_id)

    # Deduplicate within each person folder. A multi-person photo can appear
    # in several different folders by design (once per group).
    cache = {}
    for label, photo_ids in sorted(photo_groups.items()):
        folder = destination / f'Person_{label}'
        for image_id in sorted(photo_ids):
            photo, digest = unique_images[image_id]
            if copy_if_new(photo, folder, digest, cache):
                result['sorted'] += 1
            else:
                result['duplicates'] += 1
    for image_id in no_face_ids:
        photo, digest = unique_images[image_id]
        if copy_if_new(photo, destination / 'No_Faces', digest, cache):
            result['sorted'] += 1
        else:
            result['duplicates'] += 1
    result['no_faces'] = len(no_face_ids)
    return result


# ========================= WEEK 10: TKINTER GUI ============================
class PhotoSorterApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('PhotoSorter | Weeks 7–11')
        root.geometry('730x450')
        root.minsize(680, 420)
        self.source = tk.StringVar()
        self.destination = tk.StringVar()
        self.mode = tk.StringVar(value='date')
        self.threshold = tk.StringVar(value=str(DEFAULT_FACE_DISTANCE))
        self.status = tk.StringVar(value='Choose folders and a sorting mode.')
        self.messages = queue.Queue()
        frame = ttk.Frame(root, padding=20)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='PhotoSorter', font=('Arial', 22, 'bold')).pack(pady=(0, 5))
        ttk.Label(frame, text='Organise photos by date or by matching human faces').pack(pady=(0, 12))
        self.folder_row(frame, 'Source folder', self.source)
        self.folder_row(frame, 'Destination folder', self.destination)
        modes = ttk.Frame(frame)
        modes.pack(pady=(12, 5))
        ttk.Label(modes, text='Sorting mode:').pack(side='left', padx=8)
        ttk.Radiobutton(modes, text='Sort by Date', variable=self.mode, value='date').pack(side='left', padx=8)
        ttk.Radiobutton(modes, text='Sort by Face', variable=self.mode, value='face').pack(side='left', padx=8)
        settings = ttk.Frame(frame)
        settings.pack(pady=(4, 4))
        ttk.Label(settings, text='Face distance threshold (0.05–0.90):').pack(side='left')
        ttk.Entry(settings, textvariable=self.threshold, width=7).pack(side='left', padx=8)
        ttk.Label(frame, text='Face mode: use a NEW EMPTY destination folder. Larger threshold merges more faces.',
                  wraplength=650).pack(pady=(3, 5))
        self.button = ttk.Button(frame, text='Start Sorting', command=self.start)
        self.button.pack(pady=(8, 10))
        self.progress = ttk.Progressbar(frame, maximum=100, mode='determinate')
        self.progress.pack(fill='x', padx=12)
        ttk.Label(frame, textvariable=self.status, wraplength=650).pack(pady=12)
        root.after(100, self.process_messages)

    def folder_row(self, parent, label, variable):
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=5)
        ttk.Label(row, text=label, width=18).pack(side='left')
        ttk.Entry(row, textvariable=variable).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(row, text='Browse', command=lambda: self.browse(variable)).pack(side='left')

    @staticmethod
    def browse(variable):
        folder = filedialog.askdirectory()
        if folder:
            variable.set(folder)

    def start(self):
        if not self.source.get().strip() or not self.destination.get().strip():
            messagebox.showwarning('Missing folder', 'Please select both folders.')
            return
        mode = self.mode.get()
        try:
            source, destination = validate_folders(Path(self.source.get()), Path(self.destination.get()))
            threshold = float(self.threshold.get()) if mode == 'face' else DEFAULT_FACE_DISTANCE
            if mode == 'face' and not 0.05 <= threshold <= 0.90:
                raise ValueError('Face threshold must be between 0.05 and 0.90.')
            if mode == 'face' and any(destination.iterdir()):
                raise ValueError('Please select a NEW EMPTY destination folder for Face mode.')
        except (ValueError, OSError) as exc:
            messagebox.showerror('Invalid settings', str(exc))
            return
        self.button.config(state='disabled')
        self.progress['value'] = 0
        self.status.set('Processing... Face recognition can take a while on first run.')

        def progress(done, total, name):
            self.messages.put(('progress', done, total, name))

        def worker():
            try:
                if mode == 'face':
                    result = sort_photos_by_face(source, destination, progress, threshold)
                else:
                    result = sort_photos_by_date(source, destination, progress)
                self.messages.put(('complete', mode, result))
            except Exception as exc:
                logging.exception('Sorting stopped: %s', exc)
                self.messages.put(('error', str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def process_messages(self):
        try:
            while True:
                item = self.messages.get_nowait()
                if item[0] == 'progress':
                    _, done, total, name = item
                    self.progress['value'] = 100 * done / total if total else 100
                    self.status.set(f'Processing {done}/{total}: {name}')
                elif item[0] == 'complete':
                    _, mode, result = item
                    self.button.config(state='normal')
                    self.progress['value'] = 100
                    self.status.set('Finished. Photos copied: ' + str(result['sorted']))
                    lines = [f"Files found: {result['files']}",
                             f"Photos copied: {result['sorted']}",
                             f"Exact duplicates avoided: {result['duplicates']}",
                             f"Unsupported files skipped: {result['skipped']}",
                             f"Errors: {result['errors']}"]
                    if mode == 'face':
                        lines += [f"Human faces detected: {result['faces_detected']}",
                                  f"Estimated person groups: {result['people']}",
                                  f"Photos without human faces: {result['no_faces']}"]
                    else:
                        lines += [f"EXIF dates: {result['exif']}",
                                  f"File-date fallback: {result['fallback']}"]
                    if result.get('first_error'):
                        lines += ['', 'First error:', result['first_error'][:240]]
                    messagebox.showinfo('PhotoSorter results', '\n'.join(lines))
                elif item[0] == 'error':
                    self.button.config(state='normal')
                    self.status.set('Sorting failed. See error message.')
                    messagebox.showerror('PhotoSorter error', item[1])
        except queue.Empty:
            pass
        self.root.after(100, self.process_messages)


# ========================= WEEK 11: PROGRAM START ==========================
if __name__ == '__main__':
    configure_logging()
    root = tk.Tk()
    PhotoSorterApp(root)
    root.mainloop()
