#!/bin/bash

# Generate a private key
openssl genpkey -algorithm RSA -out private_key.pem -aes256

# Generate the corresponding public key
openssl rsa -pubout -in private_key.pem -out public_key.pem

echo "RSA key pair generated:"
echo "Private key: private_key.pem"
echo "Public key: public_key.pem"

