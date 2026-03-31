"use client";

import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false,loading: () => <p>Loading Chart...</p> });

export default function Chart3D() {
  const [figure, setFigure] = useState<any>(null);

  useEffect(() => {
    fetch("http://localhost:8000/graph")
      .then(res => res.json())
      .then(json => {
        const parsed = typeof json === 'string' ? JSON.parse(json) : json;
        setFigure(parsed);
      });
  }, []);
  
  if (!figure) return <p>Loading...</p>;

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
          data={figure.data}
          layout={figure.layout}
        />
       </div>
      </div>
    </main>
  );
}
