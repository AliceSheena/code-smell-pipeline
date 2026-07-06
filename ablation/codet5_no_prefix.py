from refactoring.codet5_model import CodeT5Config, CodeT5Refactor


def build_no_prefix_refactorer(model_name: str = "Salesforce/codet5-base") -> CodeT5Refactor:
    config = CodeT5Config(model_name=model_name, prefix=False)
    return CodeT5Refactor(config=config)


if __name__ == "__main__":
    refactorer = build_no_prefix_refactorer()
    example = "x = x.add_(1)\n"
    print(refactorer.generate("In-Place APIs Misused", example))
