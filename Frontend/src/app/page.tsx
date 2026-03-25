"use client";


import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';

interface PlotlyChart {
  data: any[];
  layout: any;
}

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false,loading: () => <p>Loading Chart...</p> });



export default function Home() {
const [chartData, setChartData] = useState<PlotlyChart | null>(null);

  useEffect(() => {
  fetch('/gmm_means.json')
  .then(response => response.json())
  .then(data => {setChartData(data);})
  .catch(error => console.error('Error fetching data:', error));
  }, []);

  if (!chartData) {return <div>Loading...</div>;}
  return (
    <main className="min-h-screen p-5 bg-[#F0EBE1]">

      {/* Top bar*/}
      <div className="mb-5 flex items-center">
        <h4 className="font-bold text-sm text-[#0D0D0D]">Welcome Back</h4>
        <h1 className="font-bold text-xl text-[#0D0D0D] absolute left-1/2 -translate-x-1/2">Data Visualisation</h1>

      </div>

 
      <div className="grid grid-cols-[repeat(4,1fr)] grid-rows-[repeat(4,1fr)] gap-4 h-[600px]">

      {/* Grid: auto columns and rows sized at 1fr each */}


        <div className="bg-[#E8E0D0] rounded-xl col-span-2 row-span-4 text-center flex items-center justify-center">File area</div>


      {/* Graphs*/}
        <div className="bg-[#C8B4A0] rounded-xl col-span-2 row-span-4 text-center flex items-center justify-center">
            <Plot
             data={chartData.data}
             layout={chartData.layout}
             config={{ responsive: true }}
             style={{ width: '100%', height: '100%' }}
             useResizeHandler={true}
        />
       </div>
      </div>
    </main>
  );
}