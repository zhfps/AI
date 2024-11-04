import jieba

def words_cut():
    # Sample Chinese text
    text = "我来到北京清华大学"

    # Perform word segmentation
    words = jieba.cut(text)

    # Print the result
    print("Default Mode:", " / ".join(words))

# 主函数
def main():
    words_cut()

if __name__ == "__main__":
    main()