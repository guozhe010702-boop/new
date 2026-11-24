"""Read-only final audit of real duration, IDs, resources and latency growth."""
import csv,json,statistics,math
from pathlib import Path
from control import ROOT,atomic
def main():
    folder=ROOT/'results/qualification-20261124';job=json.loads((folder/'job.json').read_text());report={'stage':job['stage'],'passed':False,'endurance':[]}
    if job['stage']!='complete':atomic(ROOT/'results/final-audit.json',report);return
    for project in ['yolo11_bytetrack','real_esrgan','depth_anything_v2']:
        path=folder/(project+'-endurance');data=list(csv.DictReader((path/'frames.csv').open()));data=[r for r in data if r.get('outcome')=='processed'];assert data
        base=int(data[0]['captured_us']);data=[r for r in data if int(r['captured_us'])-base>=5000000];ids=[int(r['id']) for r in data];times=[int(r['captured_us']) for r in data];ages=[float(r['age_ms']) for r in data];assert len(ids)>1
        resources=[json.loads(line) for line in (path/'telemetry.jsonl').read_text().splitlines() if line.strip()];stable=resources[10:-3]
        rss=[r['VmRSS']/1024 for r in stable if 'VmRSS' in r];fds=[r['fd_count'] for r in stable if 'fd_count' in r]
        width=max(1,len(data)//4);first=ages[:width];last=ages[-width:]
        def p95(v):return sorted(v)[max(0,math.ceil(.95*len(v))-1)]
        growth=p95(last)-p95(first);rss_growth=statistics.median(rss[-120:])-statistics.median(rss[:120]);fd_growth=statistics.median(fds[-120:])-statistics.median(fds[:120])
        a={'project':project,'ids_strictly_increasing':all(a<b for a,b in zip(ids,ids[1:])),'duration_s':(times[-1]-times[0])/1000000,'first_to_last_quarter_p95_growth_ms':growth,'rss_median_growth_mib':rss_growth,'fd_median_growth':fd_growth,'processed_fps':(len(data)-1)/((times[-1]-times[0])/1000000)}
        a['passed']=a['ids_strictly_increasing'] and a['duration_s']>=1780 and rss_growth<=4 and fd_growth<=2 and growth<=max(50,p95(first)*.25)
        report['endurance'].append(a)
    report['passed']=all(v['passed'] for v in report['endurance']) and all(c['pass'] for c in job['cases']);atomic(ROOT/'results/final-audit.json',report)
    if not report['passed']:raise RuntimeError('Final audit failed; preserve results and investigate before publication')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
