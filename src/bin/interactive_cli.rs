//! Interactive CLI for system analysis
//!
//! Provides a user-friendly command-line interface for analyzing systems,
//! checking model compatibility, and getting upgrade recommendations.

use anyhow::Result as AnyhowResult;
use std::io::{self, Write};
use system_analysis::*;

#[tokio::main]
async fn main() -> AnyhowResult<()> {
    // Initialize basic tracing
    tracing_subscriber::fmt::init();

    println!("🚀 System Analysis Interactive CLI");
    println!("===================================");

    let mut analyzer = SystemAnalyzer::new();
    let mut dynamic_db = DynamicModelDatabase::new();

    loop {
        println!("\nChoose an option:");
        println!("1. Analyze current system");
        println!("2. Quick system summary");
        println!("3. Check AI model compatibility");
        println!("4. Get upgrade recommendations");
        println!("5. List available models");
        println!("6. Search models (fuzzy)");
        println!("7. Model picker (smart suggestions)");
        println!("8. Model database stats");
        println!("9. Configure model fetching");
        println!("10. Refresh model database");
        println!("11. Exit");

        print!("Enter choice (1-11): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        match input.trim() {
            "1" => analyze_system(&mut analyzer).await?,
            "2" => quick_system_summary(&mut analyzer).await?,
            "3" => check_model_compatibility(&analyzer, &mut dynamic_db).await?,
            "4" => get_upgrade_recommendations(&analyzer).await?,
            "5" => list_models(&mut dynamic_db).await?,
            "6" => search_models(&mut dynamic_db).await?,
            "7" => model_picker(&mut dynamic_db).await?,
            "8" => show_database_stats(&mut dynamic_db).await?,
            "9" => configure_fetching(&mut dynamic_db).await?,
            "10" => refresh_database(&mut dynamic_db).await?,
            "11" => {
                println!("Goodbye! 👋");
                break;
            }
            _ => println!("Invalid choice. Please enter 1-11."),
        }
    }
    async fn quick_system_summary(analyzer: &mut SystemAnalyzer) -> AnyhowResult<()> {
        println!("\n📝 Quick System Summary\n=====================");
        match analyzer.quick_summary().await {
            Ok(summary) => println!("{}", summary),
            Err(e) => println!("❌ Failed to get system summary: {}", e),
        }
        Ok(())
    }

    Ok(())
}

async fn analyze_system(analyzer: &mut SystemAnalyzer) -> AnyhowResult<()> {
    println!("\n🔍 Analyzing your system...");

    let profile = analyzer
        .analyze_system()
        .await
        .map_err(|e| anyhow::anyhow!("System analysis failed: {}", e))?;

    println!("\n📊 System Analysis Results:");
    println!("==========================");
    println!("Overall Score: {:.1}/10", profile.overall_score());
    println!("CPU Score: {:.1}/10", profile.cpu_score());
    println!("GPU Score: {:.1}/10", profile.gpu_score());
    println!("Memory Score: {:.1}/10", profile.memory_score());
    println!("Storage Score: {:.1}/10", profile.storage_score());
    println!("Network Score: {:.1}/10", profile.network_score());

    if profile.npu_score > 0.0 {
        println!("NPU Score: {:.1}/10", profile.npu_score);
    }

    println!("\n💻 Hardware Summary:");
    println!("CPU: {}", profile.system_info.cpu_info.brand);
    println!(
        "Cores: {} physical, {} logical",
        profile.system_info.cpu_info.physical_cores, profile.system_info.cpu_info.logical_cores
    );
    println!(
        "Memory: {:.1} GB total, {:.1} GB available",
        profile.system_info.memory_info.total_ram as f64 / 1024.0, // Convert MB to GB
        profile.system_info.memory_info.available_ram as f64 / 1024.0
    ); // Convert MB to GB

    Ok(())
}

async fn check_model_compatibility(
    _analyzer: &SystemAnalyzer,
    dynamic_db: &mut DynamicModelDatabase,
) -> AnyhowResult<()> {
    println!("\n🤖 AI Model Compatibility Check");
    println!("===============================");

    print!("Enter model name to check (or 'list' to see available models): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let model_name = input.trim();

    if model_name == "list" {
        list_models(dynamic_db).await?;
        return Ok(());
    }

    match dynamic_db.get_model(model_name).await? {
        Some(model) => {
            println!("\n📋 Model: {}", model.name);
            println!("Family: {}", model.family);
            println!(
                "Parameters: {:.1}B",
                model.parameters as f64 / 1_000_000_000.0
            );
            println!("Memory Required: {:.1} GB", model.base_memory_gb);
            println!("Min Compute Score: {:.1}/10", model.min_compute);

            // TODO: Implement actual compatibility checking with system profile
            println!("\n✅ Compatibility analysis coming soon!");
            println!("   (Will check against your system once model compatibility is implemented)");
        }
        None => {
            println!("❌ Model '{}' not found in database.", model_name);
            println!("   Try refreshing the database or check the model name.");
        }
    }

    Ok(())
}

async fn list_models(dynamic_db: &mut DynamicModelDatabase) -> AnyhowResult<()> {
    println!("\n📚 Available AI Models");
    println!("=====================");
    println!("🔄 Loading models from database...");

    let models = dynamic_db.get_models().await?;

    if models.is_empty() {
        println!("No models found. Try refreshing the database first.");
        return Ok(());
    }

    // Group models by family
    let mut family_groups = std::collections::HashMap::new();
    for model in &models {
        family_groups
            .entry(&model.family)
            .or_insert_with(Vec::new)
            .push(model);
    }

    for (family, family_models) in family_groups {
        println!("\n🤖 {} Family:", family.to_uppercase());
        let mut sorted_models = family_models.clone();
        sorted_models.sort_by_key(|m| m.parameters);

        for model in sorted_models {
            let params_display = if model.parameters >= 1_000_000_000 {
                format!("{:.0}B", model.parameters as f64 / 1_000_000_000.0)
            } else {
                format!("{:.0}M", model.parameters as f64 / 1_000_000.0)
            };

            println!(
                "  • {} - {} params, {:.1} GB RAM",
                model.name, params_display, model.base_memory_gb
            );
        }
    }

    println!("\n💡 Total: {} models available", models.len());
    Ok(())
}

async fn search_models(dynamic_db: &mut DynamicModelDatabase) -> AnyhowResult<()> {
    println!("\n🔍 Search Models");
    println!("================");

    print!("Enter search term (model name, family, etc.): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let query = input.trim();

    if query.is_empty() {
        println!("Please enter a search term.");
        return Ok(());
    }

    println!("🔄 Searching for '{}'...", query);
    let matching_models = dynamic_db.search_models(query).await?;

    if matching_models.is_empty() {
        println!("❌ No models found matching '{}'", query);
        return Ok(());
    }

    println!("\n📋 Found {} matching models:", matching_models.len());
    for model in matching_models {
        let params_display = if model.parameters >= 1_000_000_000 {
            format!("{:.0}B", model.parameters as f64 / 1_000_000_000.0)
        } else {
            format!("{:.0}M", model.parameters as f64 / 1_000_000.0)
        };

        println!(
            "  • {} ({}) - {} params, {:.1} GB RAM",
            model.name, model.family, params_display, model.base_memory_gb
        );
    }

    Ok(())
}

async fn show_database_stats(dynamic_db: &mut DynamicModelDatabase) -> AnyhowResult<()> {
    println!("\n📊 Model Database Statistics");
    println!("============================");

    let stats = dynamic_db.get_statistics();

    println!("Total Models: {}", stats.total_models);

    if let Some(last_updated) = stats.last_updated {
        println!(
            "Last Updated: {}",
            last_updated.format("%Y-%m-%d %H:%M:%S UTC")
        );
    } else {
        println!("Last Updated: Never (using cached/curated models)");
    }

    println!("\n📈 Models by Family:");
    for (family, count) in &stats.family_counts {
        println!("  • {}: {} models", family, count);
    }

    println!("\n📏 Models by Size:");
    for (size_range, count) in &stats.size_distribution {
        println!("  • {}: {} models", size_range, count);
    }

    Ok(())
}

async fn refresh_database(dynamic_db: &mut DynamicModelDatabase) -> AnyhowResult<()> {
    println!("\n🔄 Refreshing Model Database");
    println!("============================");
    println!("Fetching latest models from Hugging Face and other sources...");
    println!("This may take a moment...");

    match dynamic_db.refresh_model_database().await {
        Ok(()) => {
            println!("✅ Database refresh completed successfully!");
            let stats = dynamic_db.get_statistics();
            println!("📊 Now tracking {} models", stats.total_models);
        }
        Err(e) => {
            println!("❌ Database refresh failed: {}", e);
            println!("💡 Using cached/curated models instead");
        }
    }

    Ok(())
}

async fn get_upgrade_recommendations(_analyzer: &SystemAnalyzer) -> AnyhowResult<()> {
    println!("\n⬆️ Upgrade Recommendations");
    println!("=========================");
    println!("Feature coming soon! Check back after model database completion.");
    Ok(())
}

async fn show_model_details(
    model_name: &str,
    dynamic_db: &mut DynamicModelDatabase,
) -> AnyhowResult<()> {
    if let Some(model) = dynamic_db.get_model(model_name).await? {
        println!("\n✅ Selected Model: {}", model.name);
        println!("📋 Details:");
        println!("   Family: {}", model.family);
        println!(
            "   Parameters: {:.1}B",
            model.parameters as f64 / 1_000_000_000.0
        );
        println!("   Memory Required: {:.1} GB", model.base_memory_gb);
        println!("   Min Compute Score: {:.1}/10", model.min_compute);
        println!("   Context Lengths: {:?}", model.context_lengths);

        print!("\nWould you like to check compatibility? (y/n): ");
        io::stdout().flush().unwrap();

        let mut confirm = String::new();
        io::stdin().read_line(&mut confirm).unwrap();

        if confirm.trim().to_lowercase() == "y" {
            println!("🔄 Compatibility checking coming soon!");
            println!("   (Will analyze against your system capabilities)");
        }
    } else {
        println!("❌ Model details not found for '{}'", model_name);
    }
    Ok(())
}

async fn show_paginated_results(
    matches: &[ModelDefinition],
    dynamic_db: &mut DynamicModelDatabase,
) -> AnyhowResult<()> {
    const PAGE_SIZE: usize = 20;
    let total_pages = matches.len().div_ceil(PAGE_SIZE);
    let mut current_page = 0;

    loop {
        let start_idx = current_page * PAGE_SIZE;
        let end_idx = std::cmp::min(start_idx + PAGE_SIZE, matches.len());
        let page_items = &matches[start_idx..end_idx];

        println!(
            "\n📋 Search Results (Page {} of {})",
            current_page + 1,
            total_pages
        );
        println!("   Showing {} models:", page_items.len());

        for (i, model) in page_items.iter().enumerate() {
            println!(
                "  {}. {} ({:.1}B params)",
                start_idx + i + 1,
                model.name,
                model.parameters as f64 / 1_000_000_000.0
            );
        }

        println!("\n💡 Commands:");
        println!("  • 1-{}: Select a model", matches.len());
        if current_page > 0 {
            println!("  • 'prev': Previous page");
        }
        if current_page + 1 < total_pages {
            println!("  • 'next': Next page");
        }
        println!("  • 'exit': Return to model picker");

        print!("\nYour choice: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "exit" {
            break;
        }

        if input == "next" && current_page + 1 < total_pages {
            current_page += 1;
            continue;
        }

        if input == "prev" && current_page > 0 {
            current_page -= 1;
            continue;
        }

        // Handle numeric selection
        if let Ok(index) = input.parse::<usize>() {
            if index > 0 && index <= matches.len() {
                show_model_details(&matches[index - 1].name, dynamic_db).await?;
                break;
            } else {
                println!("❌ Invalid selection. Choose 1-{}", matches.len());
            }
        } else {
            println!("❌ Invalid command. Try 'next', 'prev', 'exit', or a number");
        }
    }

    Ok(())
}

async fn model_picker(dynamic_db: &mut DynamicModelDatabase) -> AnyhowResult<()> {
    println!("\n🎯 Smart Model Picker");
    println!("=====================");
    println!("💡 Start typing a model name and get smart suggestions!");
    println!("   Examples: 'qwen', 'llama', 'mistral-7b', etc.");
    println!("   Commands: 'exit' to return to main menu");
    println!("            'more' to see additional results");
    println!("            'search <term>' for fuzzy search");

    loop {
        print!("\n🔍 Model search: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let query = input.trim();

        if query.is_empty() {
            continue;
        }

        if query == "exit" {
            println!("👋 Returning to main menu...");
            break;
        }

        if query == "help" {
            println!("\n📚 Help:");
            println!("  • Type part of a model name to get suggestions");
            println!("  • 'exit' - return to main menu");
            println!("  • 'more' - see more results from last search");
            println!("  • 'search <term>' - fuzzy search all models");
            println!("  • Just press Enter to try a new search");
            continue;
        }

        // Handle 'search' command for fuzzy search
        if let Some(search_term) = query.strip_prefix("search ") {
            if search_term.is_empty() {
                println!("❌ Please provide a search term after 'search'");
                continue;
            }
            println!("🔄 Fuzzy searching for '{}'...", search_term);
            let all_matches = dynamic_db.search_models(search_term).await?;
            if all_matches.is_empty() {
                println!("❌ No models found matching '{}'", search_term);
                continue;
            }
            // Show paginated results
            show_paginated_results(&all_matches, dynamic_db).await?;
            continue;
        }

        // Regular smart suggestions (limit to 10)
        println!("🔄 Finding similar models...");
        let suggestions = dynamic_db.find_similar_models(query, 10).await?;

        if suggestions.is_empty() {
            println!("❌ No models found matching '{}'", query);
            println!(
                "💡 Try 'search {}' for fuzzy search or 'help' for commands",
                query
            );
            continue;
        }

        println!("\n📋 Found {} suggestions:", suggestions.len());
        for (i, model_name) in suggestions.iter().enumerate() {
            println!("  {}. {}", i + 1, model_name);
        }

        println!("\n💡 Commands:");
        println!("  • 1-{}: Select a model", suggestions.len());
        println!("  • 'more': See additional results");
        println!("  • 'search {}': Fuzzy search all models", query);
        println!("  • 'exit': Return to main menu");
        println!("  • Enter: New search");

        print!("\nYour choice: ");
        io::stdout().flush().unwrap();

        let mut selection = String::new();
        io::stdin().read_line(&mut selection).unwrap();
        let selection = selection.trim();

        if selection.is_empty() {
            continue;
        }

        if selection == "exit" {
            println!("👋 Returning to main menu...");
            break;
        }

        if selection == "more" {
            // Get more suggestions (up to 25)
            let more_suggestions = dynamic_db.find_similar_models(query, 25).await?;
            if more_suggestions.len() > suggestions.len() {
                println!("\n📋 Extended results ({} total):", more_suggestions.len());
                for (i, model_name) in more_suggestions.iter().enumerate() {
                    println!("  {}. {}", i + 1, model_name);
                }

                print!(
                    "\nSelect a model (1-{}) or press Enter for new search: ",
                    more_suggestions.len()
                );
                io::stdout().flush().unwrap();

                let mut extended_selection = String::new();
                io::stdin().read_line(&mut extended_selection).unwrap();
                let extended_selection = extended_selection.trim();

                if !extended_selection.is_empty() {
                    if let Ok(index) = extended_selection.parse::<usize>() {
                        if index > 0 && index <= more_suggestions.len() {
                            show_model_details(&more_suggestions[index - 1], dynamic_db).await?;
                        } else {
                            println!("❌ Invalid selection");
                        }
                    } else {
                        println!("❌ Invalid input");
                    }
                }
            } else {
                println!("💡 No additional results found for '{}'", query);
            }
            continue;
        }

        // Handle 'search <term>' command
        if let Some(search_term) = selection.strip_prefix("search ") {
            if search_term.is_empty() {
                println!("❌ Please provide a search term after 'search'");
                continue;
            }
            println!("🔄 Fuzzy searching for '{}'...", search_term);
            let all_matches = dynamic_db.search_models(search_term).await?;
            if all_matches.is_empty() {
                println!("❌ No models found matching '{}'", search_term);
                continue;
            }
            // Show paginated results
            show_paginated_results(&all_matches, dynamic_db).await?;
            continue;
        }

        // Handle numeric selection
        if let Ok(index) = selection.parse::<usize>() {
            if index > 0 && index <= suggestions.len() {
                show_model_details(&suggestions[index - 1], dynamic_db).await?;
            } else {
                println!("❌ Invalid selection. Choose 1-{}", suggestions.len());
            }
        } else {
            println!("❌ Invalid input. Try:");
            println!("   • A number (1-{})", suggestions.len());
            println!("   • 'more' for additional results");
            println!("   • 'search {}' for fuzzy search", query);
            println!("   • 'exit' to return to main menu");
            println!("   • Enter for new search");
        }
    }

    Ok(())
}

async fn configure_fetching(dynamic_db: &mut DynamicModelDatabase) -> AnyhowResult<()> {
    println!("\n⚙️ Configure Model Fetching");
    println!("===========================");

    let config = dynamic_db.get_config();
    println!("📊 Current Configuration:");
    println!(
        "   Bulk Fetching: {}",
        if config.enable_bulk_fetching {
            "✅ Enabled"
        } else {
            "❌ Disabled"
        }
    );
    println!("   Min Downloads: {}", config.min_downloads_threshold);
    println!("   Max Pages: {}", config.max_pages_to_fetch);
    println!("   Models per Source: {}", config.max_models_per_source);

    println!("\n🔧 Configuration Options:");
    println!("1. Enable bulk fetching (fetch ALL models above download threshold)");
    println!("2. Disable bulk fetching (back to top 100 only)");
    println!("3. Set download threshold");
    println!("4. Set maximum pages to fetch");
    println!("5. Return to main menu");

    print!("Enter choice (1-5): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    let mut should_refresh = false;

    match input.trim() {
        "1" => {
            print!("Enter minimum downloads threshold (default 1000): ");
            io::stdout().flush().unwrap();

            let mut threshold_input = String::new();
            io::stdin().read_line(&mut threshold_input).unwrap();
            let threshold = threshold_input.trim().parse::<u64>().unwrap_or(1000);

            print!("Enter maximum pages to fetch (default 10, ~500 models): ");
            io::stdout().flush().unwrap();

            let mut pages_input = String::new();
            io::stdin().read_line(&mut pages_input).unwrap();
            let max_pages = pages_input.trim().parse::<usize>().unwrap_or(10);

            println!("🔄 Enabling bulk fetching...");
            println!("   This will fetch many more models but take longer!");
            println!("   Min downloads: {}", threshold);
            println!("   Max pages: {} (~{} models)", max_pages, max_pages * 50);

            dynamic_db.enable_bulk_fetching(threshold, max_pages);
            should_refresh = true;
            println!("✅ Configuration updated!");
        }
        "2" => {
            println!("✅ Bulk fetching disabled - using fast single-page mode");
            dynamic_db.disable_bulk_fetching();
            should_refresh = true;
            println!("✅ Configuration updated!");
        }
        "3" => {
            print!(
                "Enter new download threshold (current: {}): ",
                config.min_downloads_threshold
            );
            io::stdout().flush().unwrap();

            let mut threshold_input = String::new();
            io::stdin().read_line(&mut threshold_input).unwrap();

            if let Ok(threshold) = threshold_input.trim().parse::<u64>() {
                dynamic_db.set_download_threshold(threshold);
                should_refresh = true;
                println!("✅ Download threshold set to: {}", threshold);
                println!("✅ Configuration updated!");
            } else {
                println!("❌ Invalid number");
            }
        }
        "4" => {
            print!(
                "Enter max pages to fetch (current: {}): ",
                config.max_pages_to_fetch
            );
            io::stdout().flush().unwrap();

            let mut pages_input = String::new();
            io::stdin().read_line(&mut pages_input).unwrap();

            if let Ok(pages) = pages_input.trim().parse::<usize>() {
                dynamic_db.set_max_pages(pages);
                should_refresh = true;
                println!("✅ Max pages set to: {} (~{} models)", pages, pages * 50);
                println!("✅ Configuration updated!");
            } else {
                println!("❌ Invalid number");
            }
        }
        "5" => return Ok(()),
        _ => println!("❌ Invalid choice"),
    }

    if should_refresh {
        println!("\n� Auto-refreshing model database with new settings...");
        match dynamic_db.refresh_model_database().await {
            Ok(_) => {
                let stats = dynamic_db.get_statistics();
                println!("✅ Database refreshed successfully!");
                println!("📊 New Stats: {} total models", stats.total_models);
                if let Some(last_updated) = stats.last_updated {
                    println!(
                        "⏰ Last updated: {}",
                        last_updated.format("%Y-%m-%d %H:%M:%S UTC")
                    );
                }
            }
            Err(e) => {
                println!("⚠️ Warning: Failed to refresh database: {}", e);
                println!("💡 You can manually refresh using option 9");
            }
        }
    }

    Ok(())
}
