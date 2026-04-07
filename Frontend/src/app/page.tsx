"use client";

import FileUploader from "@/components/FileUploader";

export default function App() {

  return (
    <main className="min-h-screen p-5 bg-[#F0EBE1]">

      {/* Top bar*/}
      <div className="mb-5 flex items-center">
        <h4 className="font-bold text-sm text-[#0D0D0D]">Welcome Back</h4>
        <h1 className="font-bold text-xl text-[#0D0D0D] absolute left-1/2 -translate-x-1/2">Data Visualisation</h1>
      </div>


      <div className="grid grid-cols-[1fr_3fr] gap-4 h-[560px]">

      {/* Grid: auto columns and rows sized at 1fr each */}
        <div className="flex flex-col gap-3">
        <div className="flex-1 bg-[#E8E0D0] rounded-xl border-2 border-[#0D0D0D] flex flex-col items-center justify-center gap-2 text-[#7A7060]">
          <span className="text-sm">Upload File Here</span>
          <FileUploader/>
        </div>
        <button className="bg-[#0D0D0D] text-[#F0EBE1] py-2 px-4 rounded-lg hover:bg-[#333]">Upload File</button>
        </div>

      {/* Graphs*/}
        <div className="bg-[#C8B4A0] rounded-xl flex items-center justify-center">
          <button className="bg-[#0D0D0D] text-[#F0EBE1] py-2 px-4 rounded-lg hover:bg-[#333]">Update Graph</button>
       </div>
      </div>
    </main>
  );
}
