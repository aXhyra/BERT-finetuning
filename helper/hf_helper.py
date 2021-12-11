import subprocess
import os

from huggingface_hub import HfFolder
from huggingface_hub.constants import ENDPOINT


class HfHelper:

    def __init__(self, token=None):
        self.username = 'hf_user'
        self.token = os.environ["hf_token"] if token is None else token
        os.environ["hf_token"] = self.token
        self.hf_login()

    def write_to_credential_store(self):
        with subprocess.Popen(
                "git credential-store store".split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
        ) as process:
            input_username = f"username={self.username.lower()}"
            input_password = f"password={self.token}"

            process.stdin.write(
                f"url={ENDPOINT}\n{input_username}\n{input_password}\n\n".encode("utf-8")
            )
            process.stdin.flush()

    def hf_login(self):
        if self.token is not None:
            self.write_to_credential_store()
            HfFolder.save_token(self.token)
            print("Login successful")
            print("Your token has been saved to", HfFolder.path_token)
