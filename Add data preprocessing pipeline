</> Python
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


class DataPreprocessingPipeline:
    def __init__(self, target_column, output_dir="outputs/preprocessing"):
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = None

    def load_data(self, csv_path=None):
        if csv_path is None:
            return self._create_sample_data()

        df = pd.read_csv(csv_path)
        return df

    def _create_sample_data(self):
        np.random.seed(42)

        size = 500
        df = pd.DataFrame({
            "age": np.random.randint(18, 60, size),
            "income": np.random.normal(3500, 900, size),
            "study_time": np.random.normal(4, 1.5, size),
            "category": np.random.choice(["A", "B", "C"], size),
            "region": np.random.choice(["Seoul", "Busan", "Gwangju", "Daejeon"], size)
        })

        score = (
            df["study_time"] * 12
            + df["income"] * 0.01
            + np.where(df["category"] == "A", 8, 0)
            + np.random.normal(0, 8, size)
        )

        df["target"] = (score > score.mean()).astype(int)

        missing_index = np.random.choice(df.index, 30, replace=False)
        df.loc[missing_index, "income"] = np.nan

        return df

    def remove_outliers_iqr(self, df, columns):
        result = df.copy()

        for col in columns:
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            result[col] = np.where(
                (result[col] < lower) | (result[col] > upper),
                np.nan,
                result[col]
            )

        return result

    def build_preprocessor(self, df):
        X = df.drop(columns=[self.target_column])

        numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer([
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ])

        return numeric_features, categorical_features

    def run(self, csv_path=None):
        df = self.load_data(csv_path)

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != self.target_column]

        df = self.remove_outliers_iqr(df, numeric_cols)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        self.build_preprocessor(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        joblib.dump(self.preprocessor, self.output_dir / "preprocessor.pkl")

        pd.DataFrame({"target": y_train}).to_csv(
            self.output_dir / "y_train.csv",
            index=False
        )

        pd.DataFrame({"target": y_test}).to_csv(
            self.output_dir / "y_test.csv",
            index=False
        )

        print("전처리 완료")
        print("훈련 데이터 크기:", X_train_processed.shape)
        print("테스트 데이터 크기:", X_test_processed.shape)
        print("전처리 객체 저장:", self.output_dir / "preprocessor.pkl")


if __name__ == "__main__":
    pipeline = DataPreprocessingPipeline(target_column="target")
    pipeline.run()
