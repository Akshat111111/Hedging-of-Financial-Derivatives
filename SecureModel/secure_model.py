from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import pickle
import io

class SecureModel:
    """
    Class for encrypting and decrypting machine learning models using RSA encryption.
    """

    def __init__(self, public_key_path, private_key_path=None):
        """
        Initializes the ModelEncryptor with the provided public key and optional private key.

        Args:
            public_key_path (str): Path to the public key file.
            private_key_path (str, optional): Path to the private key file. Default is None.
        """
        # Load public key
        with open(public_key_path, 'rb') as f:
            self.public_key = RSA.import_key(f.read())
        
        #Load private key if provided
        if private_key_path:
            with open(private_key_path, 'rb') as f:
                self.private_key = RSA.import_key(f.read())
        else:
            self.private_key = None

    def _load_model(self, model):
        """
        Helper function to load the model data from different formats.

        Args:
            model (str, io.IOBase, bytes): Model data in string, file object, or bytes format.

        Returns:
            bytes: Model data in bytes format.
        """
        # If the model is a string, load it from the string
        if isinstance(model, str):
            return model
        # If the model is a file path, load it from the file
        elif isinstance(model, io.IOBase):
            return model.read()
        # If the model is already loaded in memory, return it as is
        else:
            return model

    def encrypt_model(self, model):
        """
        Encrypts the provided model using RSA encryption.

        Args:
            model (str, io.IOBase, bytes): Model data in string, file object, or bytes format.

        Returns:
            bytes: Encrypted model data.
        """
        model_data = self._load_model(model)
        model_bytes = pickle.dumps(model_data)
        cipher = PKCS1_OAEP.new(self.public_key)
        encrypted_model = cipher.encrypt(model_bytes)
        return encrypted_model

    def decrypt_model(self, encrypted_model):
        """
        Decrypts the provided encrypted model using RSA encryption.

        Args:
            encrypted_model (bytes): Encrypted model data.

        Returns:
            str: Decrypted model data.
        """
        if not self.private_key:
            raise ValueError("Private key not provided for decryption.")

        cipher = PKCS1_OAEP.new(self.private_key)
        decrypted_model_bytes = cipher.decrypt(encrypted_model)
        decrypted_model_data = pickle.loads(decrypted_model_bytes)
        return decrypted_model_data

