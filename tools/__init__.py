import tarfile
import sqlite3
from pathlib import Path


def migrate_tables(old_sqlite3_path: Path, new_sqlite3_path: Path, tables: list[str]) -> None:
    # Path 확인
    if not old_sqlite3_path.exists():
        raise FileNotFoundError(f"Source database file does not exist: {old_sqlite3_path}")

    # dst 파일이 존재하면 삭제
    if new_sqlite3_path.exists():
        new_sqlite3_path.unlink()

    # src 연결
    src_conn = sqlite3.connect(str(old_sqlite3_path))
    src_cursor = src_conn.cursor()

    # dst 연결
    dst_conn = sqlite3.connect(str(new_sqlite3_path))
    dst_cursor = dst_conn.cursor()

    # Foreign Key 옵션 활성화
    dst_cursor.execute("PRAGMA foreign_keys = ON;")
    dst_conn.commit()

    for table in tables:
        # 테이블 스키마 복사
        src_cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,))
        result = src_cursor.fetchone()
        if not result:
            print(f"Table '{table}' does not exist in source database. Skipping.")
            continue

        create_table_sql = result[0]
        dst_cursor.execute(create_table_sql)

        # 테이블 데이터 복사
        src_cursor.execute(f"SELECT * FROM {table}")
        rows = src_cursor.fetchall()

        if rows:
            # 테이블 컬럼 정보 가져오기
            src_cursor.execute(f"PRAGMA table_info({table})")
            columns_info = src_cursor.fetchall()
            column_names = [info[1] for info in columns_info]
            columns_str = ", ".join([f'"{col}"' for col in column_names])

            placeholders = ", ".join(["?"] * len(column_names))
            insert_sql = f'INSERT INTO "{table}" ({columns_str}) VALUES ({placeholders})'
            dst_cursor.executemany(insert_sql, rows)
            dst_conn.commit()

    # 커넥션 종료
    src_conn.close()
    dst_conn.close()


def compress(file_paths: list[Path], tar_gz_path: Path) -> None:
    """
    주어진 파일/디렉토리의 경로 리스트를 tar.gz로 압축합니다.

    :param file_paths: 압축할 파일 또는 디렉토리의 Path 객체 리스트
    :param tar_gz_path: 생성할 tar.gz 파일 경로(Path 객체)
    """
    tar_gz_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_gz_path, "w:gz") as tar:
        for path in file_paths:
            arcname = path.name
            tar.add(path, arcname=arcname)


def extract(tar_gz_path: Path, extract_dir: Path) -> None:
    """
    주어진 tar.gz 파일을 특정 디렉토리에 해제합니다.

    :param tar_gz_path: 압축 해제할 tar.gz 파일 경로(Path 객체)
    :param extract_dir: 압축을 풀 디렉토리(Path 객체)
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_gz_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
