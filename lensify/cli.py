import argparse, sys, json, shutil
from .search import Search, _HAS_ST, _HAS_FAISS

def doctor():
    info = {
        'sentence_transformers_available': bool(_HAS_ST),
        'faiss_available': bool(_HAS_FAISS),
    }
    return info

def main(argv=None):
    parser = argparse.ArgumentParser(prog='lensify')
    sub = parser.add_subparsers(dest='cmd')

    p_rebuild = sub.add_parser('rebuild', help='Rebuild index (force)')
    p_rebuild.add_argument('path', nargs='?', default='.')
    p_rebuild.add_argument('--show_progress', action='store_true')

    p_query = sub.add_parser('query', help='Query the index')
    p_query.add_argument('path', nargs=1)
    p_query.add_argument('q', nargs='+')
    p_query.add_argument('--k', type=int, default=6)

    p_stats = sub.add_parser('stats', help='Show index stats')
    p_stats.add_argument('path', nargs='?', default='.')

    p_info = sub.add_parser('info', help='Show info')
    p_info.add_argument('path', nargs='?', default='.')

    p_export = sub.add_parser('export', help='Export index to JSON')
    p_export.add_argument('path', nargs=2, help='[target_folder] [out.json]')

    p_doctor = sub.add_parser('doctor', help='Check environment and dependencies')

    args = parser.parse_args(argv)
    if args.cmd=='rebuild':
        s = Search(args.path)
        n = s.build(force=True, show_progress=args.show_progress)
        print(json.dumps({'indexed_chunks': n, 'path': args.path}, indent=2))
        return 0
    if args.cmd=='query':
        folder = args.path[0]
        q = ' '.join(args.q)
        s = Search(folder)
        if len(s.index.docs)==0:
            print('Index empty or missing. Building first...')
            s.build(show_progress=True)
        res = s.query(q, top_k=args.k)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.cmd=='stats' or args.cmd=='info':
        s = Search(args.path)
        print(json.dumps(s.stats(), indent=2, ensure_ascii=False))
        return 0
    if args.cmd=='export':
        folder, outp = args.path
        s = Search(folder)
        if len(s.index.docs)==0:
            s.build()
        s.export(outp)
        print(f'Exported to {outp}')
        return 0
    if args.cmd=='doctor':
        info = doctor()
        print(json.dumps(info, indent=2))
        return 0
    parser.print_help()
    return 1

if __name__=='__main__':
    raise SystemExit(main())
