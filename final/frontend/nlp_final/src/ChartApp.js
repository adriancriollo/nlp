import { Chart } from "react-google-charts";

const ChartApp = () => {
    const chartEvents = [
        {
          eventName: "select",
          callback({ chartWrapper }) {
            console.log("Selected ", chartWrapper.getChart().getSelection());
          },
        },
        {
          eventName: "ready",
          callback({ chartWrapper }) {
            console.log("Chart ready. ", chartWrapper.getChart());
          },
        },
        {
          eventName: "error",
          callback(args) {
            console.log("Chart errored. ", args);
          },
        },
      ];
      const data = [
        ["age", "weight"],
        [8, 12],
        [4, 5.5],
        [11, 14],
        [4, 5],
        [3, 3.5],
        [6.5, 7],
      ];
      const options = {
        title: "Logic Regression",
        hAxis: { title: "Y", viewWindow: { min: 0, max: 15 } },
        vAxis: { title: "X", viewWindow: { min: 0, max: 15 } },
        legend: "none",
      };
      
      return(
      <div>
        <Chart
        chartType="ScatterChart"
        data={data}
        options={options}
        chartEvents={chartEvents}
      />
      <Chart
        chartType="ScatterChart"
        data={data}
        options={options}
        chartEvents={chartEvents}
      />
      <Chart
        chartType="ScatterChart"
        data={data}
        options={options}
        chartEvents={chartEvents}
      />
      </div>);
}

export default ChartApp