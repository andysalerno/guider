class OpenChatRole:
    @staticmethod
    def role_start(role):
        if role == "user":
            return "User: "
        elif role == "assistant":
            return "Assistant: "
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return "<|end_of_turn|>"
        elif role == "assistant":
            return "<|end_of_turn|>"
        else:
            return ""


class Vicuna1_3Role:
    @staticmethod
    def role_start(role):
        if role == "user":
            return "\nUSER: "
        elif role == "assistant":
            return "\nASSISTANT: "
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return ""
        elif role == "assistant":
            return "</s>"
        else:
            return ""


class Llama2GuanacoRole:
    @staticmethod
    def role_start(role):
        if role == "user":
            return "### Human: "
        elif role == "assistant":
            return "### Assistant: "
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return ""
        elif role == "assistant":
            # return ""
            return "</s>"
        else:
            return ""


class RedmondPufferRole:
    @staticmethod
    def role_start(role):
        if role == "user":
            return "USER: "
        elif role == "assistant":
            return "ASSISTANT: "
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return ""
        elif role == "assistant":
            return "</s>"
        else:
            return ""


class Llama2ChatRole:
    @staticmethod
    def role_start(role):
        if role == "user":
            # return " <s>[INST] "
            # return " [INST] "
            return ""
        elif role == "assistant":
            return ""
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return " [/INST] "
        elif role == "assistant":
            # return " </s>"
            # return ""
            return " </s><s>[INST] "
        else:
            return ""


class Llama2UncensoredChatRole:
    @staticmethod
    def role_start(role):
        if role == "user":
            # return " <s>[INST] "
            return "### HUMAN:\n"
        elif role == "assistant":
            return "### RESPONSE:\n"
        else:
            return ""

    @staticmethod
    def role_end(role):
        if role == "user":
            return "\n"
        elif role == "assistant":
            # return " </s>"
            return "\n"
        else:
            return ""
