import subprocess
import wandb
import os

from huggingface_hub import HfFolder
from huggingface_hub.constants import ENDPOINT


class LoginHelper:

    def __init__(self,
                 hf_token=None,
                 wandb_token=None,
                 wandb_project_name="demo",
                 wandb_save_models=False,
                 tokenizer_parallelism=False):
        self.username = 'hf_user'
        try:
            self.hf_token = os.environ["hf_token"] if hf_token is None else hf_token
        except KeyError:
            print('ERROR: The Huggingface "hf_token" has not been set as environment variable nor passed to this function.')
            raise
        try:
            self.wandb_token = os.environ["wandb_token"] if wandb_token is None else wandb_token
        except KeyError:
            print('ERROR: The Wandb "wandb_token" has not been set as environment variable nor passed to this function.')
            raise

        os.environ["hf_token"] = str(self.hf_token)
        os.environ["wandb_token"] = str(self.wandb_token)
        os.environ["TOKENIZERS_PARALLELISM"] = str(tokenizer_parallelism).lower()
        self.hf_login()
        self.wandb_init(wandb_project_name, wandb_save_models)

    def write_to_credential_store(self):
        with subprocess.Popen(
                "git credential-store store".split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
        ) as process:
            input_username = f"username={self.username.lower()}"
            input_password = f"password={self.hf_token}"

            process.stdin.write(
                f"url={ENDPOINT}\n{input_username}\n{input_password}\n\n".encode("utf-8")
            )
            process.stdin.flush()

    def hf_login(self):
        if self.hf_token is not None:
            self.write_to_credential_store()
            HfFolder.save_token(self.hf_token)
            print("Login successful")
            print("Your token has been saved to", HfFolder.path_token)

    def wandb_init(self, project_name, save_models):
        if self.wandb_token is not None:
            wandb.login(key=self.wandb_token)
            wandb.init(project="ATML", entity="aXhyra")
            wandb.finish()
            os.environ["WANDB_PROJECT"] = project_name
            os.environ["WANDB_LOG_MODEL"] = str(save_models).lower()
