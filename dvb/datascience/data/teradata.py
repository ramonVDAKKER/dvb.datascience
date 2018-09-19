import decimal
import logging
import time

import pandas as pd

from ..pipe_base import PipeBase, Data, Params

try:
    import teradata
except ImportError:
    use_teradata = False
else:
    use_teradata = True

logger = logging.getLogger(__name__)

if not use_teradata:
    TeraDataImportPipe = None
else:

    class customDataTypeConverter(teradata.datatypes.DefaultDataTypeConverter):
        """
        Transforms data types from Teradata to datatypes used by Python.
        Replaces decimal comma with decimal point.
        Changes BYTEINT, BIGINT, SMALLINT and INTEGER to the Python type int.
        """

        def __init__(self):
            super().__init__(useFloat=True)

        def convertValue(self, dbType, dataType, typeCode, value):
            if value is not None and dataType == "DECIMAL":
                return decimal.Decimal(value.replace(",", "."))
            elif value is not None and (
                dataType == "BYTEINT"
                or dataType == "BIGINT"
                or dataType == "SMALLINT"
                or dataType == "INTEGER"
            ):
                return int(value)
            else:
                return super().convertValue(dbType, dataType, typeCode, value)

    class TeraDataImportPipe(PipeBase):
        """
        Reads data from Teradata and returns a dataframe.

        Args:
            file_path(str): path to read file containing SQL query
            sql(str): raw SQL query to be used

        Returns:
            A dataframe using pd.read_sql_query(), sorts the index alphabetically.
        """

        def __init__(self):
            super().__init__()
            if not use_teradata:
                logger.error("Teradata module is not imported and could not be used")

        def transform(self, data: Data, params: Params) -> Data:
            if params["file_path"]:
                with open(params["file_path"], "r") as f:
                    sql = f.read()
            else:
                sql = params["sql"]
            sort_alphabetically = params["sort_alphabetically"]

            start = time.time()
            udaExec = teradata.UdaExec(
                appName="td",
                version="1.0",
                configureLogging=False,
                logConsole=False,
                logLevel="TRACE",
                dataTypeConverter=customDataTypeConverter(),
            )
            conn = udaExec.connect(method="odbc", DSN="Teradata")
            df = pd.read_sql_query(sql, conn)
            logger.info(
                "teradata returned %s rows in % seconds",
                str(len(df)),
                str(round(time.time() - start)),
            )
            conn.close()

            if sort_alphabetically:
                df = df.sort_index(axis=1)

            return {"df": df}
