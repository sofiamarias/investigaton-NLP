from Analysis import Analysis
from Analysis3_latencia import Analysis3

def main():
    a = Analysis()
    a.make_bar_chart_wrong_answers_by_types()

    b = Analysis3()
    b.make_latency_chart()

if __name__ == "__main__":
    main()

