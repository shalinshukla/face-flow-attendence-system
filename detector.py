import argparse
import pickle
from typing import List, Any

import face_recognition
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# Create directories if they don't already exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)
args = parser.parse_args()


def encode_known_faces(
        model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """
    This function takes care of encoding known faces from images in a training directory.
    It uses the face_recognition library to detect and encode faces in images.
    Parameters:
    - model (str): The face detection model to use, default is "hog".
    - encodings_location (Path): The location where the face encodings will be saved.
    Returns:
    - None
    """
    # Lists to store names and corresponding face encodings
    names = []
    encodings = []
    # Iterate through all image files in the "training" directory
    for filepath in Path("training").glob("*/*"):
        # Extract the person's name from the directory structure
        name = filepath.parent.name

        # Load the image using the face_recognition library
        image = face_recognition.load_image_file(filepath)

        # Detect face locations and calculate face encodings
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Store the person's name and their face encodings
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
    # Create a dictionary with names and corresponding face encodings
    name_encodings = {"names": names, "encodings": encodings}
    # Save the dictionary to a binary file using the pickle module
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> list[Any]:
    """
    Given an unknown image, get the locations and encodings of any faces and
    compares them against the known encodings to find potential matches.
    """
    names = []
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        names.append(name)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()
    return names





def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown face encoding and a list of known face encodings,
    this function finds the known face encoding with the most matches.

    Parameters:
    - unknown_encoding: The encoding of the unknown face to be recognized.
    - loaded_encodings: A dictionary containing two lists - 'names' and 'encodings',
      representing the names of known faces and their corresponding encodings.

    Returns:
    - The name of the recognized face.
    """

    # Compare the unknown face encoding with all known face encodings
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding, tolerance=0.5
    )
    # Count the votes for each known face based on matches
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )

    # If there are votes, return the name of the recognized face with the most votes
    if votes:
        return votes.most_common(1)[0][0]


def _display_face(draw, bounding_box, name):
    """
    Draws bounding boxes around faces, a caption area, and text captions.
    """
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill=BOUNDING_BOX_COLOR,
        outline=BOUNDING_BOX_COLOR,
    )
    draw.text(
        (text_left, text_top),
        name,
        fill=TEXT_COLOR,
    )


def validate(model: str = "hog"):
    """
    Runs recognize_faces on a set of images with known faces to validate
    known encodings.
    """
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
