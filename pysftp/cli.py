from .core import Pool, async_main
import asyncio

def cli():
    import sys
    if not len(sys.argv) == 4:
        print("use as pysftp user@server remotesrc destination")
        sys.exit(0)

    username, server =  sys.argv[1].split('@')
    pattern = sys.argv[2]
    dst = sys.argv[3]
    pool = Pool(server, username=username)
    asyncio.run(async_main(pattern, dst, pool))

if __name__ == '__main__':
    cli()
