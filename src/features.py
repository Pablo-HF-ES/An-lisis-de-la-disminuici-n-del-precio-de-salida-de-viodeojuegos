import pandas as pd
import os
import joblib
print("Versión de Joblib:", joblib.__version__)
import sklearn
print("Versión de scikit-learn:", sklearn.__version__)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


def build_features(input_path, output_dir, models_dir):
    """
    Construye características a partir de los datos intermedios y guarda los resultados en la carpeta de características.
    """
    try:
        print(f"Iniciando construcción de características desde: {input_path}")
        
        # 1. Leer los datos intermedios
        data = pd.read_csv(input_path)

        # 2. Sustituir valores nulos por 0
        data.fillna(value = 0, inplace = True)

        # 3. Eliminar filas duplicadas
        data.drop_duplicates(inplace = True)

        data = data.select_dtypes(exclude = ['object'])

        X = data.drop(columns=['launch_price_decay'])
        y = data['launch_price_decay']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        preprocessing_pipeline = Pipeline([('escalador', StandardScaler()), ('seleccionador', SelectKBest(k = 5, score_func = f_regression))])

        preprocessing_pipeline.fit(X_train, y_train)
        X_train_transformed = preprocessing_pipeline.transform(X_train)
        X_test_transformed = preprocessing_pipeline.transform(X_test)

        # Guardamos el pipeline completo
        joblib.dump(preprocessing_pipeline, os.path.join(models_dir, 'preprocessing_pipeline.joblib'))

        # 4. Asegurar que la carpeta de destino exista antes de guardar
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # 5. Convertimos los arrays de vuelta a DataFrame para poder usar .to_csv()
        pd.DataFrame(X_train_transformed).to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        pd.DataFrame(X_test_transformed).to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        
        # Guardamos las 'y'
        y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

        print(f"\nLa construcción de características se ha completado con éxito.")
        print(f"Características guardadas en: {output_dir}")

    except Exception as e:
        print(f"Un error ocurrió durante la construcción de características: {e}")

if __name__ == "__main__":
    RUTA_ENTRADA = 'C:\\Users\\phf14_dcq7b60\\Desktop\\Video_Games ML Project\\data\\interim\\vg_price_forecasting_interim.csv'
    RUTA_SALIDA = 'C:\\Users\\phf14_dcq7b60\\Desktop\\Video_Games ML Project\\data\\processed\\vg_price_forecasting_definitive.csv'
    RUTA_MODELOS = 'C:\\Users\\phf14_dcq7b60\\Desktop\\Video_Games ML Project\\models\\'

    build_features(RUTA_ENTRADA, RUTA_SALIDA, RUTA_MODELOS)