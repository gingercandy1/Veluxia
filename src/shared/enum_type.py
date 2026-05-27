import enum


class FactoryType(enum.Enum):
    Text = 1
    Image = 2
    ImageFrame = 3,
    Animation = 4,
    Speech = 5

    Translation = 6

    @staticmethod
    def convert_by_text(text: str):
        if text == "text":
            return FactoryType.Text
        elif text == "image":
            return FactoryType.Image
        elif text == "animation":
            return FactoryType.Animation
        elif text == "image_frame":
            return FactoryType.ImageFrame
        elif text == "speech":
            return FactoryType.Speech
        return None

    @staticmethod
    def convert_to_text(_type: enum.Enum):
        if _type == FactoryType.Text:
            return "text"
        elif _type == FactoryType.Image:
            return "image"
        elif _type == FactoryType.ImageFrame:
            return "image_frame"
        elif _type == FactoryType.Animation:
            return "animation"
        elif _type == FactoryType.Speech:
            return "speech"
        return None