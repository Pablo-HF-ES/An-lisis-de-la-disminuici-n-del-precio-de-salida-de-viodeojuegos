import pandas as pd
print("Versión de Pandas:", pd.__version__)
import os
print("Versión de OS:", os.name)

def data_ingestion(input_path, output_path):
    
    """
    Ingiere datos desde un archivo CSV, realiza un recorte básico 
    y guarda los datos en la carpeta de datos intermedios.
    """
    try:
        print(f"Iniciando lectura desde: {input_path}")
        
        # 1. Pandas lee el CSV directamente (reemplaza a open().read())
        data = pd.read_csv(input_path)

        # 2. Exploración básica
        print(f"\nInformación del dataset: {data.info()}")
        print(f"\nEstadísticas básicas: {data.describe()}")
        print(f'Tipo de datos de cada columna: {data.dtypes}')
        print(f"Número de filas y columnas: {data.shape}")

        # 3. Procesamiento básico (recorte)
        processed_data = data.iloc[1:500000]
    
        # 4. Asegurar que la carpeta de destino exista antes de guardar
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 5. Pandas guarda el CSV directamente (reemplaza a open().write())
        # index=False evita que se guarde la columna de números de fila
        processed_data.to_csv(output_path, index=False)

        print(f"\nLa ingesta de datos se ha completado con éxito.")
        print(f"Datos guardados en: {output_path}")

    except Exception as e:
        print(f"Un error ocurrió durante la ingesta de datos: {e}")


if __name__ == "__main__":
    RUTA_ENTRADA = 'C:\\Users\\phf14_dcq7b60\\Desktop\\Video_Games ML Project\\data\\raw\\vg_price_forecasting.csv'
    RUTA_SALIDA = 'C:\\Users\\phf14_dcq7b60\\Desktop\\Video_Games ML Project\\data\\interim\\vg_price_forecasting_interim.csv'
    
    data_ingestion(RUTA_ENTRADA, RUTA_SALIDA)