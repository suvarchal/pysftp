import asyncio
import time
from os.path import sep

import asyncssh
import tqdm

use_uvloop = True

try:
    if use_uvloop:
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    print(' is not available, using default loop')


def random_name(num=5, starts_with=""):
    import string
    import random
    num = num if starts_with == "" else num - 1
    return starts_with + "".join(random.choice(string.ascii_letters) for _ in range(num))


def parse_src_format(src_path):
    src_path_split = src_path.split(os.sep)
    post_fix = []
    pre_fix = []
    parse_it = False
    dirname = random_name(3, "d")
    for i, x in enumerate(src_path_split[:-1]):  # is len(pa_split) == 1 skip
        if not parse_it:
            if "*" in x:
                parse_it = True
                prefix_ind = i
        if parse_it:
            post_fix.append("{" + f"{dirname}{i:05d}" + "}") # 5 is max supported TODO 
        else:
            pre_fix.append(x)

    parse_format = os.sep.join([*pre_fix, *post_fix, "{file}"])
    # print(pre_fix, post_fix)
    # dst_path = dst_path + os.sep.join(*post_fix)
    # print(dst_path)
    # os.makedirs(dst_path, exist_ok=True)
    return parse_format


class ProgressMonitor(object):
    """" We and also use this to learn drops in speedd etc, adjust bytesyze etc smartly"""

    def __init__(self):
        self.current_dict = dict()
        self.downloading = []
        self._total_files = 0
        self._total_bytes = 0.0
        self._avg_speed = 0.0
        self.start_time = time.time()

    @property
    def total_files(self):
        return self._total_files

    @total_files.setter
    def total_files(self, num):
        self._total_files += num

    @property
    def avg_speed(self):
        return self._total_bytes / (time.time() - self.start_time)

    def progress_handler(self, src, dst, current_bytes, total_bytes):
        src_fi = src.decode('ascii')

        if src_fi in self.current_dict:

            if current_bytes == total_bytes:
                completed_tqdm = self.current_dict.pop(src_fi)
                completed_tqdm.update(current_bytes - completed_tqdm.prev_bytes)
                completed_tqdm.refresh()
                completed_tqdm.close()  # leave?
                self._total_files += 1
                self._total_bytes += total_bytes
            else:
                # tqdm.tqdm.write(f"{current}, {total}")
                x = self.current_dict[src_fi]
                x.update(current_bytes - x.prev_bytes)  # this kind of update is better then using block size
                # because we can use option to update progress sparingly (miniters)
                x.prev_bytes = current_bytes

        else:  # initalize tqdm for a downloading object
            desc = src_fi.split(sep)[-1]
            self.current_dict[src_fi] = tqdm.tqdm(total=total_bytes, desc=desc, disable=None, unit='B', unit_scale=True,
                                                  unit_divisor=1024, ascii=True, miniters=1)  # , leave=False)
            # miniters decides how often we call progress
            self.current_dict[src_fi].prev_bytes = 0.0  # is this necessary


async def run_stat(path, pool):
    # conn_context = await pool.acquire()
    # async with conn_context as conn:
    #     async with conn.start_sftp_client() as sftp:
    #         res = await sftp.stat(path)
    #         #print(res)
    conn_client = await pool.acquire(start_client=True)
    async with conn_client as sftp:
        res = await sftp.stat(path)
    return res


def hr_size(size):
    """Returns human readable size
       input size in bytes
    """
    units = ["B", "kB", "MB", "GB", "TB", "PB"]
    i = 0  # index
    while size > 1024 and i < len(units):
        i += 1
        size /= 1024.0
    return f"{size:.2f} {units[i]}"


async def file_download(path, dst, pool, progress_handler=None):
    conn_client = await pool.acquire(start_client=True)
    async with conn_client as sftp:
        res = await sftp.get(path, localpath=dst, progress_handler=progress_handler)  # , block_size=131072)

def make_dst_dir(src_format, src, dst):
    import parse
    local = [dst]
    p = parse.parse(src_format, src)
    ex = [p.named[k] for k in sorted(p.named.keys())[:-1]] # -1 to skip filename
    local.extend(ex)
    dst = os.sep.join(local)
    os.makedirs(dst, exist_ok=True)
    return dst

async def entry_point(pattern, dst, pool):
    conn_client = await pool.acquire(start_client=True)  # why cant this be written in same line
    async with conn_client as sftp:
        res = await sftp.glob(pattern)

    monitor = ProgressMonitor()
    # tasks = (asyncio.create_task(file_download(fi, pool, progress_handler=monitor.progress_handler)) for fi in
    #         res)  # best option
    # tasks can be used to give names, etc.

    src_formatter = parse_src_format(pattern)
    tasks = [file_download(fi, make_dst_dir(src_formatter, fi, dst), pool, progress_handler=monitor.progress_handler) for fi in res]  # best option
    await asyncio.gather(*tasks)
    # asyncio swaps corps in tasks any ways
    # tasks can have a name and a callback that is cool to give coarse grained status without using progeress handler
    # for t in asyncio.as_completed(tasks):
    #    await t
    # tqdm.tqdm.write(f"Downloaded: {monitor.total_files} files, Avg. Speed:  {monitor.avg_speed / 1024} kB/s")
    # tqdm.tqdm.write(f"Downloaded: {monitor.total_files} files, Avg. Speed:  {hr_size(monitor.avg_speed)}/s")
    return f"\nDownloaded: {monitor.total_files} files, Avg. Speed:  {hr_size(monitor.avg_speed)}/s, " \
           f"Total time: {time.time() - monitor.start_time:.2f} seconds"


class ConnectionHolder(object):
    def __init__(self, pool):
        self.pool = pool  # parent pool
        self.conn = None
        self.started = False
        self.is_active = False
        self.sftp_client = None  # optional checking if we can reuse this

    async def start(self, sftp_client=False):
        self.conn = await asyncssh.connect(*self.pool.connection_args, **self.pool.connection_kwargs)
        self.started = True
        if sftp_client:
            self.sftp_client = await self.conn.start_sftp_client()

    async def close(self):
        if self.conn is not None:
            await self.conn.close()


class ConnectionContext(object):
    def __init__(self, pool, start_client=False):
        self.pool = pool
        self.conn_holder = None
        self.start_client = start_client

    async def __aenter__(self):
        self.conn_holder = await self.pool.queue.get()  # waits till there is available connection , get_nowait() will not.
        if not self.conn_holder.started:
            await self.conn_holder.start(self.start_client)
            # print('started a connection')

        ### also  we can add timeout's  or if connection is closed we could start again
        ###  how to check if connection is closed
        if self.start_client:
            return self.conn_holder.sftp_client

        return self.conn_holder.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ nothing async about it otherwise we need to implement __await__"""
        self.pool.queue.put_nowait(self.conn_holder)  # put connection back to the pool


## class ConnectionHolder()
## """ instead of adding connection in queue if we add a holder
##     we use a class to hold connection we can add few metrics like
##     if connection is being used or not, also may be speed of connection etc.
##     also can be used to start a new connection when needed
##     also add call back


class Pool(object):
    """ idea is to create a queue with resuable connection
    pools, possibly dynamically genrate new connection based on number of downloads"""

    def __init__(self, *args, **kwargs):
        self.pool_size = kwargs.pop('pool_size', 8)
        self.connection_args = args
        self.connection_kwargs = kwargs
        self.queue = asyncio.LifoQueue()  # important because released connections are put in the end
        self.connection_list = []  # used to close in the end
        self._async__init__()  # not sure how to model this

    def _async__init__(self):
        """not really async"""
        for _ in range(self.pool_size):
            ch = ConnectionHolder(self)
            self.connection_list.append(ch)
            self.queue.put_nowait(ch)

    async def acquire(self, start_client=False):
        # start_client should probably be class attribute
        return ConnectionContext(self, start_client=start_client)

    def close(self):
        for c in self.connection_list:
            try:
                if c.conn is not None:
                    c.conn.close()
            except Exception as ex:
                # print(ex)
                pass

    async def __aenter__(self):
        pass

    async def __aexit__(self, *exc_args):
        pass
    #      self._async__init__()
    #
    # def __await__(self):
    #     return self._async__init__().__await__()


import os


async def async_main(pattern_g, dst, pool_g):
    if not os.path.exists(dst):
        raise Exception(f"Desitination directory {dst} doesn't exist.")
    return_str = await entry_point(pattern=pattern_g, dst=dst, pool=pool_g)
    tqdm.tqdm.write(return_str)
    pool_g.close()


# start2 = time.time()

if __name__ == '__main__':
    import sys

    username, server = sys.argv[1].split("@")
    pattern = sys.argv[2]
    try:
        pool = Pool(server, username=username)
        asyncio.get_event_loop().run_until_complete(async_main(pattern, pool))
    except (OSError, asyncssh.Error) as exc:
        sys.exit('SFTP operation failed what?: ' + str(exc))

# print("End:", time.time(), start2)
# print(f"total time: {time.time() - start2} seconds", flush=True)
# 2
