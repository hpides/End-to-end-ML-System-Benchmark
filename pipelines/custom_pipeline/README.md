# End to End Benchmarking

This script lets you benachmark all kinds of commands using umlaut.
"<your command>" should be the command you would execute in a terminal like
```python3 dummy.py"```
It should also include all flags your process needs.

The folder argument should be the absolute path to where your script is located.

```
python3 run_script.py -cmd "<your command>" -folder "<path/to/your/script>" -g -gm -gt -gp -t -c -m
```