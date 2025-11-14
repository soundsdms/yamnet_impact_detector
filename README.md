# yamnet_impact_detector

a tool to automate inserting sound effects at moments of impact in a sound file, based on yamnet classification.

## Impact detection CLI

The first building block of the project is a command line utility that analyses the
audio track of a video file and reports timestamps where impact-like events were
found.  The detector uses [Google's YAMNet](https://tfhub.dev/google/yamnet/1)
model via TensorFlow Hub.

```bash
python -m yamnet_impact_detector.cli input_video.mp4 --threshold 0.55 --output impacts.json
```

The command above loads `input_video.mp4`, extracts its audio track, and
classifies each frame with YAMNet.  Detected impacts (timestamp and confidence)
are written to `impacts.json`.  If `--output` is omitted the detections are
printed to stdout.

### Configuration options

* `--threshold` – probability threshold that must be exceeded before an impact
  is emitted.
* `--smoothing` – size of the moving-average window applied to the impact score
  time-series, in seconds.
* `--min-gap` – minimum spacing between impact detections, in seconds.  This can
  be used to merge very rapid sequences of detections.
* `--keywords` – override the default set of class-name keywords that are used
  to decide which YAMNet classes are impact related.

## Installation

Install the Python dependencies into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow and the audio extraction helper rely on external system libraries.
Ensure that `libsndfile` (used by librosa) and `ffmpeg` are installed and
available on your `PATH`.  Refer to the respective project documentation if you
encounter import errors.

## Next steps

* Integrate the detector with a pipeline that overlays sound effects at the
  detected timestamps.
* Add tools for evaluating the detector on a labelled impact dataset.
