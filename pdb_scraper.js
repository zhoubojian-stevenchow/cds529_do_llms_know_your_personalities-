async function scrapeDirectIDsV9() {
    // --- CONFIGURATION ---
    const START_ID = 1;
    const END_ID = 10; // The profile ID where your scraping ends
    const CONCURRENCY_LIMIT = 500; // Reduced to 5 to stop network congestion
    const CHUNK_SIZE = 1000;     // Saving more often to be safe
    
    // --- STATE ---
    window.uniqueProfiles = new Map();
    window.fileCounter = 1;
    window.isSaving = false;
    
    let queue = [];
    for (let i = START_ID; i <= END_ID; i++) {
        queue.push(i);
    }
    
    console.log(`🚀 Starting V9 "CORS FIX" Scraper (Range: ${START_ID} - ${END_ID})`);

    // --- WORKER ---
    async function worker(id) {
        while (queue.length > 0) {
            let profileId = queue.shift(); 
            
            if (profileId % 50 === 0) {
                console.log(`   [Worker ${id}] Checking ID #${profileId} (Found: ${window.uniqueProfiles.size})`);
            }

            let profileData = await fetchProfileByID(profileId);
            
            if (profileData) {
                console.log(`   ✅ FOUND: ${profileData.mbti_profile || profileData.name} (ID: ${profileData.id})`);
                window.uniqueProfiles.set(profileData.id, profileData);
            }

            // SAVE CHECK
            if (window.uniqueProfiles.size >= CHUNK_SIZE && !window.isSaving) {
                await downloadMap(window.uniqueProfiles, window.fileCounter);
            }
            
            // Delay to prevent network clogging
            await new Promise(r => setTimeout(r, 300)); 
        }
    }

    // --- API CALL (FIXED HEADERS) ---
    async function fetchProfileByID(id) {
        try {
            // Using the V1 endpoint that we know exists
            let url = `https://api.personality-database.com/api/v1/profile/${id}`;
            
            let res = await fetch(url, {
                "headers": {
                    "accept": "application/json, text/plain, */*",
                    "accept-language": "zh-CN,zh;q=0.9",
                    // "priority": "u=1, i", <--- REMOVED: This caused the block
                    "sec-fetch-dest": "empty",
                    "sec-fetch-mode": "cors",
                    "sec-fetch-site": "same-site"
                },
                "referrer": "https://www.personality-database.com/",
                "body": null,
                "method": "GET",
                "mode": "cors",
                "credentials": "include" 
            });

            if (res.status === 404) return null; 
            
            if (!res.ok) {
                console.warn(`⚠️ Error on ID ${id}: Status ${res.status}`);
                return null;
            }
            
            let json = await res.json();
            return json.id ? json : (json.data || null);
            
        } catch (e) {
            // Silence network errors to keep console clean
            return null;
        }
    }

    // --- SAVE FUNCTION ---
    async function downloadMap(mapData, part) {
        window.isSaving = true;
        console.log(`💾 SAVING Part ${part} (${mapData.size} items)...`);
        
        try {
            let arrayData = Array.from(mapData.values());
            let jsonString = JSON.stringify(arrayData, null, 2);
            let blob = new Blob([jsonString], { type: "application/json" });
            let url = URL.createObjectURL(blob);
            let a = document.createElement('a');
            a.href = url;
            a.download = `pdb_v9_dump_part${part}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            window.uniqueProfiles.clear();
            window.fileCounter++;
            await new Promise(r => setTimeout(r, 3000));
        } catch (e) {
            console.error("Save failed", e);
        } finally {
            window.isSaving = false;
        }
    }

    // --- LAUNCH ---
    let promises = [];
    for (let i = 0; i < CONCURRENCY_LIMIT; i++) {
        promises.push(worker(i + 1));
    }
    await Promise.all(promises);
    
    if (window.uniqueProfiles.size > 0) {
        downloadMap(window.uniqueProfiles, window.fileCounter);
    }
    console.log("🎉 SCRAPE COMPLETE.");
}
