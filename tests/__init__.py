""" In terminal 1 first we run the local host api by the input [cd ~/Desktop/self-rag
source .venv/bin/activate
uvicorn api.main:app --reload]
after that we run the terminal 2 by the code [cd ~/Desktop/self-rag
source .venv/bin/activate
python eval/run_benchmark.py]
by this we get the data of self_rag and baseline as average case of many (8) questions which we put in the run_benhmark.py """

"""If we want to find answer of a single seperately for self_rag as well as baseline then the terminal 1 code[make api] and
   for terminal 2 code [curl -X POST http://127.0.0.1:8000/ask\
                        -H "Content -Type:application/json"\
                        -d '{"question":"type your question here?","mode":"self_rag"}'] for self_rag result
                        OR
                        [curl -X POST http://127.0.0.1:8000/ask\
                        -H "Content -Type:application/json"\
                        -d '{"question":"type your question here?","mode":"baseline"}'] for baseline result """

"""If we want to add new pdf then the terminal comands
   [cd ~/Desktop/self-rag
source .venv/bin/activate

# 1. Put PDFs in data/raw/

# 2. Rebuild the index
python -m src.ingest --reset

# 3. Start API
uvicorn api.main:app --reload]
then test [curl http://127.0.0.1:8000/health]"""

"""After replacing json file with new json file then the command to check for that the new json file works
   without any reitrate the problem base from previoius json file [sed -n '1,240p' eval/run_benchmark.py] """
