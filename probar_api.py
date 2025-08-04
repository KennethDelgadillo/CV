import requests

# URL del backend (modifícalo si tu API usa otro puerto o dominio)
url = "http://127.0.0.1:5000/api/upload"

# Especifica las rutas de los archivos que deseas enviar
cv_path = "CV Kenneth.pdf"  # Ruta del archivo CV
job_description_path = "testjp2.txt"  # Ruta del archivo con la descripción del trabajo

# Abre los archivos para enviarlos como datos en la solicitud
files = {
    "cv": open(cv_path, "rb"),  # Archivo del CV
    "job_position": open(job_description_path, "rb")  # Archivo de la descripción del trabajo
}

# Enviar solicitud POST al backend con los archivos
try:
    print("Enviando archivos al backend...")
    response = requests.post(url, files=files)

    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        print("Respuesta exitosa del backend")
        # Imprimir el JSON devuelto por el backend
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    print("Ocurrió un error al intentar conectarse al backend:")
    print(e)

finally:
    # Asegúrate de cerrar los archivos abiertos
    files["cv"].close()
    files["job_position"].close()