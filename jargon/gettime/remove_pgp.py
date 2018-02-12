ifn = "dark_texts_withpgp.txt"
ofn = "dark_texts.txt"

a = " real data ---begin pgp message=== end pgp message--- sadfww4ywuw349t0==="


def remove_between(s, start, end):
    res = list()
    for x in s.split(start):
        res.append(x.split(end, 1)[-1])

    return " ".join(res)


with open(ifn) as ifd:
    with open(ofn, 'w') as ofd:
        for line in ifd:
            if "begin pgp public key block" in line:
                line = remove_between(line, "begin pgp public key block", " end pgp public key block")
            if "begin pgp private key block" in line:
                line = remove_between(line, "begin pgp private key block", " end pgp private key block")
            if "begin pgp message" in line:
                line = remove_between(line, "begin pgp message", " end pgp message")
            ofd.write(line.strip())
            ofd.write("\n")
