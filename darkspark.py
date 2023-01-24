"""
DarkSpark Searchable Symmetric Encryption library
==================================================

[Searchable symmetric encryption](SSE)
* SSE is a symmetric-key encryption scheme that allows one to search a collection of [encrypted documents](ED) without the ability to decrypt them.
* - Encrypt a collection of documents D=(D_1, ..., D_n)
* - Each document D_i \subseteq W is viewed as a set of keywords from a [keyword space](W).
* - Given the [encryption key](K) and a keyword w \in W, the SSE generates a [search token](tk) with which the ED can be searched for keyword w.
* - The result of the search is the  subset of ED which contains the keyword w.

A static SSE scheme consists of three algorithms (SETUP, TOKEN, SEARCH) that work as follows:

* SETUP takes as input a security parameter k and a document collection D and outputs a symmetric key K and an encrypted document collection ED.
* TOKEN takes as input the symmetric key K and a keyword w and outputs a search token tk.
* SEARCH takes as input the encrypted document collection ED and a search token tk and outputs a set of encrypted documents R \subseteq ED.

The SSE scheme is used by a client and an untrusted server as follows.
* The client encrypts its data collection using the SETUP algorithm which returns a secret key K and an encrypted document collection ED.
* The client keeps K secret and sends ED to the untrusted server.
* To search for a keyword w, the client runs the SEARCH TOKEN algorithms on K and w to generate a search token tk which it sends to the server.
* The server runs SEARCH with ED and tk and returns the resulting encrypted documents back to the client.
"""

import secrets
import json
from typing import List, Tuple, Iterable
from dataclasses import dataclass, field
from enum import Enum
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding, hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.exceptions import InvalidSignature


class KeywordSpace(Enum):
    """Keyword space W"""

    ALPHABET = "abcdefghijklmnopqrstuvwxyz"

    @classmethod
    def get_random_keyword(cls) -> str:
        """Return a random keyword from the keyword space W"""

        return secrets.choice(cls.ALPHABET)

    @classmethod
    def get_random_keywords(cls, n: int) -> List[str]:
        """Return a list of n random keywords from the keyword space W"""

        return [cls.get_random_keyword() for _ in range(n)]


@dataclass
class Document:
    """Document D_i \subseteq W"""

    keywords: List[str] = field(default_factory=KeywordSpace.get_random_keywords)

    def __post_init__(self):
        """Ensure that all keywords are in the keyword space W"""

        for keyword in self.keywords:
            if keyword not in KeywordSpace.ALPHABET:
                raise ValueError(f"{keyword} is not in the keyword space")

    def __repr__(self):
        return f"Document({self.keywords})"

    def __str__(self):
        return f"{self.keywords}"


@dataclass
class DocumentCollection:
    """Document collection D=(D_1, ..., D_n)"""

    documents: List[Document] = field(default_factory=list)

    def __post_init__(self):
        """Ensure that all documents are unique"""

        if len(self.documents) != len(set(self.documents)):
            raise ValueError("All documents must be unique")

    def __repr__(self):
        return f"DocumentCollection({self.documents})"

    def __str__(self):
        return f"{self.documents}"


@dataclass
class EncryptedDocument:
    """Encrypted document ED_i"""

    ciphertext: bytes = field(default_factory=bytes)

    def __init__(self, ciphertext: bytes, iv: bytes):
        """Initialize an encrypted document with ciphertext and iv."""
        self._ciphertext = ciphertext
        self._iv = iv

    def __post_init__(self):
        """Ensure that the ciphertext is not empty"""

        if not self.ciphertext:
            raise ValueError("Ciphertext cannot be empty")

    def __repr__(self):
        return f"EncryptedDocument({self.ciphertext})"

    def __str__(self):
        return f"{self.ciphertext}"

    @property
    def ciphertext(self) -> bytes:
        """Return the ciphertext of the encrypted document."""
        return self._ciphertext

    @property
    def iv(self) -> bytes:
        """Return the initialization vector of the encrypted document."""
        return self._iv

    @classmethod
    def from_json(cls, json_string: str) -> "EncryptedDocument":
        """Return an encrypted document from a json string."""
        d = json.loads(json_string)
        return cls(bytes.fromhex(d["ciphertext"]), bytes.fromhex(d["iv"]))

    def to_json(self) -> str:
        """Return a json string representation of the encrypted document."""
        return json.dumps({"ciphertext": self._ciphertext.hex(), "iv": self._iv.hex()})


@dataclass
class EncryptedDocumentCollection:
    """Encrypted document collection ED=(ED_1, ..., ED_n)"""

    encrypted_documents: List[EncryptedDocument] = field(default_factory=list)

    def __post_init__(self):
        """Ensure that all encrypted documents are unique"""

        if len(self.encrypted_documents) != len(set(self.encrypted_documents)):
            raise ValueError("All encrypted documents must be unique")

    def __len__(self) -> int:
        """Return the number of encrypted documents in the collection."""
        return len(self.encrypted_documents)

    def __iter__(self) -> Iterable[EncryptedDocument]:
        """Return an iterator over the encrypted documents in the collection."""
        return iter(self.encrypted_documents)

    def __repr__(self) -> str:
        """Return a string representation of the encrypted document collection."""
        return f"EncryptedDocumentCollection({list(self)})"

    def __str__(self) -> str:
        """Return a string representation of the encrypted document collection."""
        return f"EncryptedDocumentCollection({list(self)})"

    @classmethod
    def from_json(cls, json_string: str) -> "EncryptedDocumentCollection":
        """Return an encrypted document collection from a json string."""
        d = json.loads(json_string)
        return cls([EncryptedDocument.from_json(json.dumps(doc)) for doc in d["docs"]])

    def to_json(self) -> str:
        """Return a json string representation of the encrypted document collection."""
        return json.dumps({"docs": [doc.to_json() for doc in self]})


class Encryption:
    """A class that represents a symmetric-key encryption scheme."""

    def __init__(self, key: bytes):
        """Initialize an encryption scheme with a key."""
        self._key = key

    def encrypt(self, document: Document) -> EncryptedDocument:
        """Return an encrypted document."""
        raise NotImplementedError

    def decrypt(self, encrypted_document: EncryptedDocument) -> Document:
        """Return a decrypted document."""
        raise NotImplementedError


class AESEncryption(Encryption):
    """A class that represents an AES encryption scheme."""

    def __init__(self, key: bytes):
        """Initialize an AES encryption scheme with a key."""
        super().__init__(key)
        self._backend = default_backend()
        self._cipher = Cipher(
            algorithms.AES(self._key), modes.CBC(b"\x00" * 16), self._backend
        )

    def encrypt(self, document: Document) -> EncryptedDocument:
        """Return an encrypted document."""
        encryptor = self._cipher.encryptor()
        padder = padding.PKCS7(128).padder()
        padded_data = (
            padder.update(json.dumps(document.keywords).encode()) + padder.finalize()
        )
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        return EncryptedDocument(ciphertext, encryptor.iv)

    def decrypt(self, encrypted_document: EncryptedDocument) -> Document:
        """Return a decrypted document."""
        decryptor = self._cipher.decryptor()
        unpadder = padding.PKCS7(128).unpadder()
        padded_data = (
            decryptor.update(encrypted_document.ciphertext) + decryptor.finalize()
        )
        data = unpadder.update(padded_data) + unpadder.finalize()
        return Document(json.loads(data))


@dataclass
class SearchToken:
    """Search token tk"""

    token: bytes = field(default_factory=bytes)

    def __post_init__(self):
        """Ensure that the search token is not empty"""

        if not self.token:
            raise ValueError("Search token cannot be empty")

    def __repr__(self):
        return f"SearchToken({self.token})"

    def __str__(self):
        return f"{self.token}"

    @classmethod
    def from_json(cls, json_string: str) -> "SearchToken":
        """Return a search token from a json string."""
        d = json.loads(json_string)
        return cls(bytes.fromhex(d["token"]))

    def to_json(self) -> str:
        """Return a json string representation of the search token."""
        return json.dumps({"token": self.token.hex()})


class SearchTokenGenerator:
    """A class that represents a search token generator."""

    def __init__(self, key: bytes):
        """Initialize a search token generator with a key."""
        self._key = key

    def generate_search_token(self, keyword: str) -> SearchToken:
        """Return a search token for a keyword."""
        raise NotImplementedError


class AESSearchTokenGenerator(SearchTokenGenerator):
    """A class that represents an AES search token generator."""

    def __init__(self, key: bytes):
        """Initialize an AES search token generator with a key."""
        super().__init__(key)
        self._backend = default_backend()
        self._cipher = Cipher(
            algorithms.AES(self._key), modes.CBC(b"\x00" * 16), self._backend
        )

    def generate_search_token(self, keyword: str) -> SearchToken:
        """Return a search token for a keyword."""
        encryptor = self._cipher.encryptor()
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(keyword.encode()) + padder.finalize()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        return SearchToken(ciphertext)


class Search:
    """A class that represents a search scheme."""

    def __init__(self, key: bytes):
        """Initialize a search scheme with a key."""
        self._key = key

    def search(
        self,
        encrypted_document_collection: EncryptedDocumentCollection,
        search_token: SearchToken,
    ) -> List[EncryptedDocument]:
        """Return a list of encrypted documents that contain the keyword."""
        raise NotImplementedError


class AESSearch(Search):
    """A class that represents an AES search scheme."""

    def __init__(self, key: bytes):
        """Initialize an AES search scheme with a key."""
        super().__init__(key)
        self._backend = default_backend()
        self._cipher = Cipher(
            algorithms.AES(self._key), modes.CBC(b"\x00" * 16), self._backend
        )

    def search(
        self,
        encrypted_document_collection: EncryptedDocumentCollection,
        search_token: SearchToken,
    ) -> List[EncryptedDocument]:
        """Return a list of encrypted documents that contain the keyword."""
        decryptor = self._cipher.decryptor()
        unpadder = padding.PKCS7(128).unpadder()
        padded_data = decryptor.update(search_token.token) + decryptor.finalize()
        keyword = unpadder.update(padded_data) + unpadder.finalize()
        return [
            doc
            for doc in encrypted_document_collection
            if keyword in self._decrypt(doc)
        ]

    def _decrypt(self, encrypted_document: EncryptedDocument) -> Document:
        """Return a decrypted document."""
        decryptor = self._cipher.decryptor()
        unpadder = padding.PKCS7(128).unpadder()
        padded_data = (
            decryptor.update(encrypted_document.ciphertext) + decryptor.finalize()
        )
        data = unpadder.update(padded_data) + unpadder.finalize()
        return Document(json.loads(data))


class SigningKey:
    """A class that represents a signing key."""

    def __init__(self, private_key: bytes, public_key: bytes):
        """Initialize a signing key with a private key and a public key."""
        self._private_key = private_key
        self._public_key = public_key

    def __repr__(self) -> str:
        """Return a string representation of the signing key."""
        return f"SigningKey({self._private_key}, {self._public_key})"

    def __str__(self) -> str:
        """Return a string representation of the signing key."""
        return f"SigningKey({self._private_key}, {self._public_key})"

    @property
    def private_key(self) -> bytes:
        """Return the private key of the signing key."""
        return self._private_key

    @property
    def public_key(self) -> bytes:
        """Return the public key of the signing key."""
        return self._public_key

    @classmethod
    def generate(cls) -> "SigningKey":
        """Return a new signing key."""
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        public_key = private_key.public_key()
        return cls(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ),
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ),
        )

    @classmethod
    def from_pem(cls, private_pem: str, public_pem: str) -> "SigningKey":
        """Return a signing key from a private pem and a public pem."""
        private_key = serialization.load_pem_private_key(
            private_pem.encode(), password=None, backend=default_backend()
        )
        public_key = serialization.load_pem_public_key(
            public_pem.encode(), backend=default_backend()
        )
        return cls(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            ),
            public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            ),
        )

    def to_pem(self) -> Tuple[str, str]:
        """Return a private pem and a public pem of the signing key."""
        return (self._private_key.decode(), self._public_key.decode())


class Signing:
    """A class that represents a signing scheme."""

    def __init__(self, signing_key: SigningKey):
        """Initialize a signing scheme with a signing key."""
        self._signing_key = signing_key

    def sign(self, data: bytes) -> bytes:
        """Return a signature of the data."""
        raise NotImplementedError

    def verify(self, data: bytes, signature: bytes) -> bool:
        """Return True if the signature is valid for the data."""
        raise NotImplementedError


class RSASigning(Signing):
    """A class that represents an RSA signing scheme."""

    def __init__(self, signing_key: SigningKey):
        """Initialize an RSA signing scheme with a signing key."""
        super().__init__(signing_key)
        self._private_key = serialization.load_pem_private_key(
            self._signing_key.private_key, password=None, backend=default_backend()
        )
        self._public_key = serialization.load_pem_public_key(
            self._signing_key.public_key, backend=default_backend()
        )

    def sign(self, data: bytes) -> bytes:
        """Return a signature of the data."""
        return self._private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

    def verify(self, data: bytes, signature: bytes) -> bool:
        """Return True if the signature is valid for the data."""
        try:
            self._public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except InvalidSignature:
            return False


class KeyDerivationFunction:
    """Key derivation function"""

    def __init__(self, salt: bytes):
        """Initialize the key derivation function"""

        self.salt = salt

    def derive(self, password: bytes, length: int) -> bytes:
        """Derive a key from the password using the key derivation function"""

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=length,
            salt=self.salt,
            iterations=100000,
            backend=default_backend(),
        )
        key = kdf.derive(password)

        return key


class PublicKeyInfrastructure:
    """Public key infrastructure"""

    def __init__(self, private_key: rsa.RSAPrivateKey, public_key: rsa.RSAPublicKey):
        """Initialize the public key infrastructure"""

        self.private_key = private_key
        self.public_key = public_key

    def sign(self, message: bytes) -> bytes:
        """Sign the message using the public key infrastructure"""

        signature = self.private_key.sign(
            message,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return signature

    def verify(self, message: bytes, signature: bytes) -> bool:
        """Verify the signature of the message using the public key infrastructure"""

        try:
            self.public_key.verify(
                signature,
                message,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except InvalidSignature:
            return False


class SSE:
    """A class that represents a static searchable symmetric encryption scheme."""

    def __init__(
        self,
        encryption: Encryption,
        search_token_generator: SearchTokenGenerator,
        search: Search,
    ):
        """Initialize a static searchable symmetric encryption scheme with an encryption scheme, a search token generator, and a search scheme."""
        self._encryption = encryption
        self._search_token_generator = search_token_generator
        self._search = search

    def setup(
        self, document_collection: List[Document]
    ) -> Tuple[bytes, EncryptedDocumentCollection]:
        """Return a secret key and an encrypted document collection."""
        key = secrets.token_bytes(32)
        return (
            key,
            EncryptedDocumentCollection(
                [self._encryption.encrypt(doc) for doc in document_collection]
            ),
        )

    def token(self, key: bytes, keyword: str) -> SearchToken:
        """Return a search token for a keyword."""
        return self._search_token_generator.generate_search_token(keyword)

    def search(
        self,
        encrypted_document_collection: EncryptedDocumentCollection,
        search_token: SearchToken,
    ) -> List[EncryptedDocument]:
        """Return a list of encrypted documents that contain the keyword."""
        return self._search.search(encrypted_document_collection, search_token)


class EncryptedSSE:
    """A class that represents an encrypted searchable symmetric encryption scheme"""

    def __init__(
        self,
        sse: SSE,
        key_derivation_function: KeyDerivationFunction,
        signing: Signing,
        public_key_infrastructure: PublicKeyInfrastructure,
    ):
        """Initialize an encrypted searchable symmetric encryption scheme with an encryption scheme, a search token generator, and a search scheme."""

        self._encryption = sse._encryption
        self._search_token_generator = sse._search_token_generator
        self._search = sse._search
        self._key_derivation_function = key_derivation_function
        self._signing = signing
        self._public_key_infrastructure = public_key_infrastructure

    def setup(self, password: bytes, document_collection: List[Document]) -> str:
        """Return a json string representation of the encrypted document collection."""

        # Generate a shared secret based on the password and salt.
        shared_secret = (
            self._key_derivation_function.derive(password=password, length=32)
            if password is not None
            else secrets.token_bytes(32)
        )

        # Generate a secret key based on the shared secret.
        secret_key = self._key_derivation_function.derive(
            password=shared_secret, length=32
        )

        # Create an encryption scheme based on the secret key.
        encryption = AESEncryption(key=secret_key)

        # Create an encrypted document collection based on the document collection and encryption scheme.
        encrypted_document_collection = EncryptedDocumentCollection(
            [encryption.encrypt(doc) for doc in document_collection]
        )

        # Sign the encrypted document collection.
        signature = self._signing.sign(encrypted_document_collection.to_json().encode())

        # Create a dictionary representation of the setup.
        setup = {
            "encrypted_document_collection": encrypted_document_collection,
            "signature": signature,
        }

        return json.dumps(setup)

    def token(self, password: bytes, keyword: str) -> str:
        """Return a json string representation of the search token."""

        if password is not None:

            # Generate a shared secret based on the password and salt.
            shared_secret = self._key_derivation_function.derive(
                password=password, length=32
            )

            # Generate a secret key based on the shared secret.
            secret_key = self._key_derivation_function.derive(
                password=shared_secret, length=32
            )

            # Create a search token generator based on the secret key.
            search_token_generator = AESSearchTokenGenerator(key=secret_key)

        else:

            # Generate a secret key.
            secret_key = secrets.token_bytes(32)

            # Create a search token generator based on the secret key.
            search_token_generator = AESSearchTokenGenerator(key=secret_key)

        # Generate a search token for the keyword using the search token generator.
        search_token = search_token_generator.generate_search_token(keyword)

        return json.dumps({"search_token": search_token})

    def search(self, password: bytes, token: str, encrypted_documents: str) -> str:
        """Return a json string representation of the list of encrypted documents that contain the keyword."""

        if password is not None:

            # Generate a shared secret based on the password and salt.
            shared_secret = self._key_derivation_function.derive(
                password=password, length=32
            )

            # Generate a secret key based on the shared secret.
            secret_key = self._key_derivation_function.derive(
                password=shared_secret, length=32
            )

            # Create a search scheme based on the secret key.
            search = AESSearch(key=secret_key)

        else:

            # Generate a secret key.
            secret_key = secrets.token_bytes(32)

            # Create a search scheme based on the secret key.
            search = AESSearch(key=secret_key)

        # Create an encrypted document collection from the json string representation of the encrypted documents.
        encrypted_document_collection = EncryptedDocumentCollection.from_json(
            encrypted_documents
        )

        # Create a search token from the json string representation of the search token.
        search_token = SearchToken.from_json(token)

        # Perform the search using the search scheme and return the json string representation of the list of encrypted documents that contain the keyword.
        return json.dumps(
            {
                "encrypted_documents": list(
                    search.search(encrypted_document_collection, search_token)
                )
            }
        )

    def verify(self, encrypted_documents: str, signature: bytes) -> bool:
        """Return True if the signature is valid for the encrypted documents."""

        # Create an encrypted document collection from the json string representation of the encrypted documents.
        encrypted_document_collection = EncryptedDocumentCollection.from_json(
            encrypted_documents
        )

        # Verify the signature of the encrypted documents using the public key infrastructure.
        return self._public_key_infrastructure.verify(
            encrypted_document_collection.to_json().encode(), signature
        )
