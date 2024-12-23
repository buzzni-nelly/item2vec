import pathlib

from item2vec.volume import Volume


def main():

    def clear_directory(directory_path: pathlib.Path):
        """재귀적으로 디렉토리 내 모든 파일 및 하위 디렉토리를 삭제합니다."""
        for x in directory_path.iterdir():
            if x.is_file():
                x.unlink()
            elif x.is_dir():
                clear_directory(x)
        directory_path.rmdir()

    volume_c = Volume(company_id="aboutpet", model="carca", version="v1")

    workspace_path = volume_c.workspace_path
    if workspace_path.exists() and workspace_path.is_dir():
        for item in workspace_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                clear_directory(item)


if __name__ == "__main__":
    main()
