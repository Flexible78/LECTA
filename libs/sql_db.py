import sqlite3
import logging
from typing import Dict, List, Optional, Any

class SQLiteDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._connect()

    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # для доступа по имени
        except sqlite3.Error as e:
            logging.error(f"Failed to connect to database {self.db_path}: {e}")
            raise

    def select(self, db_name: str, colls: Dict[str, Any], args: dict = {}) -> List[List[Any]]:
        cursor = self.conn.cursor()
        COLUMNS = ', '.join(colls.keys())
        where_conditions = []
        where_values = []
        for key, value in colls.items():
            if value:
                where_conditions.append(f"{key} = ?")
                where_values.append(value)
        WHERE = 'WHERE ' + ' AND '.join(where_conditions) if where_conditions else ''
        SORT = f"ORDER BY {args['sort']} ASC" if 'sort' in args else 'ORDER BY 1 ASC'
        query = f"SELECT {COLUMNS} FROM {db_name} {WHERE} {SORT}"
        try:
            cursor.execute(query, where_values)
            result = [list(row) for row in cursor.fetchall()]
            logging.debug(f"SELECT executed: {query}, params: {where_values}, found: {len(result)}")
            return result
        except sqlite3.Error as e:
            logging.error(f"SELECT error from {db_name}: {e}")
            return []

    def delete(self, table: str, condition: Dict[str, Any]) -> bool:
        cursor = self.conn.cursor()
        key = list(condition.keys())[0]
        value = condition[key]
        try:
            cursor.execute(f"DELETE FROM {table} WHERE {key} = ?", (value,))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logging.error(f"DELETE error from table {table}: {e}")
            return False

    def upsert(self, table: str, data: Dict[str, Any]) -> bool:
        cursor = self.conn.cursor()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        values = list(data.values())
        try:
            cursor.execute(
                f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})",
                values
            )
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logging.error(f"INSERT/REPLACE error into table {table}: {e}")
            return False

    def close(self):
        if self.conn:
            self.conn.close()

sql_db = SQLiteDatabase('tts.db')