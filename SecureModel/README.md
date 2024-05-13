## Steps to Secure you AI/ML model:-

1. Generate RSA public-private keypair by running generate_keypair.sh
    a. In bash or shell, run ```chmod +x generate_keypair.sh```
    b. After that execute ```./generate_keypair.sh```
2. Enter any pass key with which you want to decrypt your private_key.pem
3. Re-Enter the same pass key to generate two files private_key.pem and public_key.pem
4. After that decrypt the private key whenever you want to use it by running ```openssl rsa -in private_key.pem -out decrypted_private_key.pem```
5. This will decrypt the private_key.pem and store it in decrypted_private_key.pem
6. Now import secure_model and make an object of SecureModel class with providing path to public_key.pem and decrypted_private_key.pem(optional)
7. Use ```_load_model()``` function to load your saved model in any format 
8. Use ```encrypt_model()``` function to encrypt your model
9. Use ```decrypt_model()``` function to decrypt your model


